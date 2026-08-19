from __future__ import annotations

import importlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms


def _as_bool(value: Union[bool, str, int]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


def _flatten_visuals(value: Any) -> Iterable[Any]:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_visuals(item)
        return
    yield value


def _report_progress(completed: int, total: int, task: str, group_finished: bool = False) -> None:
    """Atomically merge authoritative adapter progress into the queue state."""
    task = os.environ.get("SHOWO2_PROGRESS_TASK", task)
    raw_path = os.environ.get("SHOWO2_PROGRESS_FILE")
    if not raw_path:
        return
    if not task:
        return
    try:
        path = Path(raw_path)
        state = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        completed_before = int(os.environ.get("SHOWO2_PROGRESS_COMPLETED_BEFORE", "0"))
        overall_total = int(os.environ.get("SHOWO2_PROGRESS_OVERALL_TOTAL", str(total)))
        task_offset = int(state.get("task_completed_offset", 0))
        current_done = task_offset + completed
        task_total = int(state.get("expected_samples", total))
        state.update(
            {
                "current_task": task,
                "expected_samples": task_total,
                "completed_samples": current_done,
                "task_percent": round(100.0 * current_done / task_total, 2) if task_total else 100.0,
                "overall_percent": round(100.0 * (completed_before + current_done) / overall_total, 2)
                if overall_total
                else 100.0,
                "pid": os.getpid(),
                "updated_at": int(time.time()),
            }
        )
        if group_finished:
            state["task_completed_offset"] = current_done
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    except Exception as exc:
        warnings.warn(f"Could not update Show-o2 progress file: {exc}", RuntimeWarning)


class ShowO2Qwen25(lmms):
    """lmms-eval adapter for Show-o2 Qwen2.5 image understanding.

    The production path follows ``inference_mmu.py``: Wan VAE latents are
    encoded by both Show-o2 vision branches, fused, and decoded by the shared
    Qwen backbone. ``dry_run`` deliberately skips all model/weight imports and
    exists only for installation and request-contract checks.
    """

    is_simple = True

    def __init__(
        self,
        showo2_root: Optional[str] = None,
        config_file: Optional[str] = None,
        pretrained: Optional[str] = None,
        vae_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: Union[int, str] = 1,
        weight_type: str = "bfloat16",
        vae_weight_type: str = "float32",
        max_new_tokens: Union[int, str] = 300,
        top_k: Union[int, str] = 1,
        temperature: Union[float, str] = 1.0,
        seed: Union[int, str] = 0,
        system_prompt: str = "You are a helpful assistant.",
        dry_run: Union[bool, str, int] = False,
        dry_run_response: str = "SHOWO2_DRY_RUN_OK",
        allow_cpu: Union[bool, str, int] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown Show-o2 model arguments: {unknown}")

        inferred_root = Path(__file__).resolve().parents[3]
        self.showo2_root = Path(showo2_root or os.environ.get("SHOWO2_ROOT", inferred_root)).expanduser().resolve()
        self.config_file = Path(
            config_file
            or os.environ.get("SHOWO2_CONFIG", self.showo2_root / "configs/showo2_1.5b_demo_432x432.yaml")
        ).expanduser().resolve()
        self.pretrained = pretrained or os.environ.get("SHOWO2_MODEL")
        self.vae_path_override = vae_path or os.environ.get("SHOWO2_VAE_PATH")
        self.requested_device = str(device)
        self.batch_size_per_gpu = int(batch_size)
        self.weight_type = self._parse_dtype(weight_type)
        self.vae_weight_type = self._parse_dtype(vae_weight_type)
        self.max_new_tokens = int(max_new_tokens)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.system_prompt = system_prompt.replace("\\n", "\n")
        self.dry_run = _as_bool(dry_run)
        self.dry_run_response = str(dry_run_response)
        self.allow_cpu = _as_bool(allow_cpu)

        if self.batch_size_per_gpu != 1:
            raise ValueError("Show-o2 MMU adapter currently supports only batch_size=1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

        self._device = torch.device("cpu")
        if self.dry_run:
            logger.info("Initialized Show-o2 adapter in dry-run mode; no weights were loaded")
            return

        self._load_model()

    @staticmethod
    def _parse_dtype(value: str) -> torch.dtype:
        normalized = str(value).lower().replace("torch.", "")
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype {value!r}; use bfloat16, float16, or float32") from exc

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer(self):
        if self.dry_run:
            return None
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def _select_device(self) -> None:
        from accelerate import Accelerator

        self.accelerator = Accelerator()
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        requested = self.requested_device.lower()
        wants_cuda = requested == "auto" or requested.startswith("cuda")
        if wants_cuda and torch.cuda.is_available():
            if self._world_size > 1:
                self._device = self.accelerator.device
            elif requested in {"auto", "cuda"}:
                self._device = torch.device("cuda:0")
            else:
                self._device = torch.device(self.requested_device)
            return

        if wants_cuda and not torch.cuda.is_available():
            if not self.allow_cpu:
                raise RuntimeError(
                    "CUDA is unavailable. Show-o2 real inference requires a GPU; "
                    "use dry_run=true for contract smoke tests or allow_cpu=true only for debugging."
                )
            warnings.warn("Running Show-o2 on CPU is unsupported and will be extremely slow", RuntimeWarning)
            self._device = torch.device("cpu")
            return

        self._device = torch.device(self.requested_device)
        if self._device.type == "cpu" and not self.allow_cpu:
            raise RuntimeError("device=cpu requires allow_cpu=true because real Show-o2 inference is GPU-oriented")

    def _ensure_showo2_imports(self):
        if not (self.showo2_root / "models/__init__.py").is_file():
            raise FileNotFoundError(f"Show-o2 source root is invalid: {self.showo2_root}")

        root_string = str(self.showo2_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)

        existing = sys.modules.get("models")
        if existing is not None:
            module_file = Path(getattr(existing, "__file__", "")).resolve()
            if self.showo2_root not in module_file.parents:
                raise ImportError(
                    "The generic Python package name 'models' was already imported from "
                    f"{module_file}; start lmms-eval in a clean process so Show-o2 can import its own models package."
                )

        showo_models = importlib.import_module("models")
        misc = importlib.import_module("models.misc")
        return showo_models, misc

    def _load_model(self) -> None:
        if not self.config_file.is_file():
            raise FileNotFoundError(f"Show-o2 config was not found: {self.config_file}")
        self._select_device()

        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError("omegaconf is required for the Show-o2 adapter") from exc

        showo_models, misc = self._ensure_showo2_imports()
        config = OmegaConf.load(self.config_file)
        OmegaConf.resolve(config)
        if config.model.vae_model.type != "wan21":
            raise NotImplementedError(f"Unsupported Show-o2 VAE type: {config.model.vae_model.type}")

        if self.pretrained:
            config.model.showo.pretrained_model_path = self.pretrained
        self.pretrained = str(config.model.showo.pretrained_model_path)

        raw_vae_path = self.vae_path_override or str(config.model.vae_model.pretrained_model_path)
        vae_path = Path(raw_vae_path).expanduser()
        if not vae_path.is_absolute():
            vae_path = self.showo2_root / vae_path
        self.vae_path = vae_path.resolve()
        if not self.vae_path.is_file():
            raise FileNotFoundError(
                f"Wan VAE checkpoint was not found: {self.vae_path}. Set vae_path=... or SHOWO2_VAE_PATH."
            )

        llm_path = str(config.model.showo.llm_model_path)
        self._tokenizer, self.showo_token_ids = misc.get_text_tokenizer(
            llm_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name="qwen2_5",
        )
        config.model.showo.llm_vocab_size = len(self._tokenizer)

        logger.info(f"Loading Wan VAE from {self.vae_path}")
        self.vae_model = showo_models.WanVAE(
            vae_pth=str(self.vae_path),
            dtype=self.vae_weight_type,
            device=self._device,
        )
        logger.info(f"Loading Show-o2 from {self.pretrained}")
        if config.model.showo.load_from_showo:
            self._model = showo_models.Showo2Qwen2_5.from_pretrained(
                self.pretrained,
                use_safetensors=False,
            )
        else:
            raise NotImplementedError("The adapter currently requires model.showo.load_from_showo=true")
        self._model = self._model.to(device=self._device, dtype=self.weight_type).eval()
        self._model.requires_grad_(False)

        self.config = config
        self.resolution = int(config.dataset.preprocessing.resolution)
        self.num_mmu_image_tokens = int(config.dataset.preprocessing.num_mmu_image_tokens)
        if bool(config.model.showo.add_time_embeds):
            self.num_mmu_image_tokens += 1

        self.sys_prompt_ids = self._tokenizer(
            f"system\n{self.system_prompt}<|im_end|>",
            add_special_tokens=False,
        ).input_ids
        self.role_user_ids = self._tokenizer(
            "\n<|im_start|>user\n",
            add_special_tokens=False,
        ).input_ids
        self.role_assistant_ids = self._tokenizer(
            "\n<|im_start|>assistant\n",
            add_special_tokens=False,
        ).input_ids
        self.omni_attn_mask_naive = showo_models.omni_attn_mask_naive
        logger.info(f"Show-o2 adapter ready on {self._device} (rank {self.rank}/{self.world_size})")

    def _resolve_doc(self, request: Instance, task: str, split: str, doc_id: Any) -> Any:
        request_doc = getattr(request, "doc", None)
        if request_doc is not None:
            return request_doc
        try:
            return self.task_dict[task][split][doc_id]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(f"Could not resolve document {doc_id!r} for {task}/{split}") from exc

    def _resolve_image(
        self,
        request: Instance,
        doc_to_visual,
        doc_id: Any,
        task: str,
        split: str,
    ) -> Image.Image:
        doc = self._resolve_doc(request, task, split, doc_id)
        raw_visuals = doc_to_visual(doc)
        visuals = list(_flatten_visuals(raw_visuals))
        if not visuals:
            raise ValueError(f"Show-o2 requires one image, but {task}/{split}/{doc_id} supplied none")
        if len(visuals) > 1:
            warnings.warn(
                f"Show-o2 adapter supports one image per request; using the first of {len(visuals)} images",
                RuntimeWarning,
            )

        visual = visuals[0]
        if isinstance(visual, Image.Image):
            return visual.convert("RGB")
        if isinstance(visual, (str, os.PathLike)):
            path = Path(visual)
            if path.suffix.lower() in {".mp4", ".avi", ".mov", ".webm"}:
                raise NotImplementedError("The Show-o2 lmms-eval adapter currently supports images, not video")
            with Image.open(path) as image:
                return image.convert("RGB")
        raise TypeError(f"Unsupported visual type for Show-o2: {type(visual).__name__}")

    def _image_transform(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True),
            ]
        )
        return transform(image)

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @torch.inference_mode()
    def _answer(self, context: str, image: Image.Image, gen_kwargs: dict[str, Any]) -> str:
        self._set_seed()
        context = context.replace("<image>", "").strip()
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = float(gen_kwargs.get("temperature", self.temperature))
        do_sample_value = gen_kwargs.get("do_sample")
        do_sample = temperature > 0 if do_sample_value is None else _as_bool(do_sample_value)
        top_k_value = gen_kwargs.get("top_k", self.top_k)
        top_k = None if top_k_value is None else int(top_k_value)
        if temperature < 0 or (do_sample and temperature == 0):
            raise ValueError("temperature must be positive when sampling and non-negative for greedy decoding")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        if top_k == 0:
            top_k = None

        image_tensor = self._image_transform(image).unsqueeze(0).to(self._device)
        image_latents = self.vae_model.sample(image_tensor.unsqueeze(2)).squeeze(2).to(self.weight_type)
        image_embeds_und = self._model.image_embedder_und(image_latents)
        image_embeds_gen = self._model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + self._model.position_embedding(self._model.image_position_ids)
        image_embeds_und = self._model.und_trans(image_embeds_und)["last_hidden_state"]
        image_embeds = self._model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        input_ids = self._tokenizer(context, add_special_tokens=False).input_ids
        prefix_ids = [self.showo_token_ids["bos_id"], *self.sys_prompt_ids, *self.role_user_ids]
        suffix_ids = [self.showo_token_ids["boi_id"], self.showo_token_ids["eoi_id"], *input_ids, *self.role_assistant_ids]
        text_tokens_a = torch.tensor(prefix_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        text_tokens_b = torch.tensor(suffix_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        text_embeds_a = self._model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self._model.showo.model.embed_tokens(text_tokens_b)

        if bool(self.config.model.showo.add_time_embeds):
            time_value = torch.tensor([[1.0]], device=self._device)
            time_embeds = self._model.time_embed(time_value, text_embeds_a.dtype)
            if hasattr(self._model, "time_embed_proj"):
                time_embeds = self._model.time_embed_proj(time_embeds)
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
                dim=1,
            ).to(self.weight_type)
            image_offset = text_tokens_a.shape[1] + 2
        else:
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]],
                dim=1,
            ).to(self.weight_type)
            image_offset = text_tokens_a.shape[1] + 1

        modality_positions = torch.tensor(
            [[[image_offset, self.num_mmu_image_tokens]]],
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = self.omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=self._device,
            inverted=True,
        ).to(input_embeds.dtype)

        tokens = self._model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=top_k,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            eos_token=self._tokenizer.eos_token_id,
        )
        if not tokens:
            return ""
        output_ids = torch.stack(tokens).reshape(1, -1)
        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _apply_until(text: str, gen_kwargs: dict[str, Any]) -> str:
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        if until is None:
            until = []
        if not isinstance(until, (list, tuple)):
            raise TypeError("gen_kwargs['until'] must be a string or a list of strings")
        stops = [text.find(stop) for stop in until if stop and text.find(stop) >= 0]
        return text[: min(stops)] if stops else text

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses: List[str] = []
        progress_task = requests[0].task_name if requests else ""
        _report_progress(0, len(requests), progress_task)
        for request in requests:
            try:
                context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            except ValueError as exc:
                raise ValueError(
                    "Show-o2 expects lmms-eval generate_until arguments "
                    "(context, gen_kwargs, doc_to_visual, doc_id, task, split)"
                ) from exc
            gen_kwargs = dict(gen_kwargs or {})
            image = self._resolve_image(request, doc_to_visual, doc_id, task, split)
            if self.dry_run:
                response = self.dry_run_response
            else:
                response = self._answer(str(context), image, gen_kwargs)
            response = self._apply_until(response, gen_kwargs)
            responses.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            _report_progress(len(responses), len(requests), task)
        _report_progress(len(responses), len(requests), progress_task, group_finished=True)
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Show-o2 adapter does not implement loglikelihood; use generation-based tasks")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Show-o2 adapter currently supports single-round image understanding only")


class ShowO2Qwen25Video(ShowO2Qwen25):
    """Show-o2 multi-frame video adapter following inference_mmu_vid.py."""

    VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        *args: Any,
        num_video_frames: Union[int, str] = 32,
        video_token_size: Union[int, str] = 14,
        **kwargs: Any,
    ) -> None:
        self.num_video_frames = int(num_video_frames)
        self.video_token_size = int(video_token_size)
        if self.num_video_frames < 1:
            raise ValueError("num_video_frames must be positive")
        if self.video_token_size < 1:
            raise ValueError("video_token_size must be positive")
        super().__init__(*args, **kwargs)

    @staticmethod
    def _uniform_sample(frames: List[Image.Image], count: int) -> List[Image.Image]:
        if not frames:
            raise ValueError("The video contains no decodable frames")
        indices = np.linspace(0, len(frames) - 1, count, dtype=int)
        return [frames[int(index)].convert("RGB") for index in indices]

    def _decode_video(self, path: Path) -> List[Image.Image]:
        try:
            import av
        except ImportError as exc:
            raise ImportError("PyAV is required for Show-o2 video evaluation; install the av package") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Video file was not found: {path}")
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream was found in {path}")
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)
            if total_frames > 0:
                targets = set(
                    int(index)
                    for index in np.linspace(0, total_frames - 1, self.num_video_frames, dtype=int)
                )
                decoded = {
                    index: frame.to_image().convert("RGB")
                    for index, frame in enumerate(container.decode(stream))
                    if index in targets
                }
                available = [decoded[index] for index in sorted(decoded)]
                if available:
                    return self._uniform_sample(available, self.num_video_frames)
            else:
                frames = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
                return self._uniform_sample(frames, self.num_video_frames)
        raise ValueError(f"No video frames could be decoded from {path}")

    def _resolve_video(
        self,
        request: Instance,
        doc_to_visual,
        doc_id: Any,
        task: str,
        split: str,
    ) -> List[Image.Image]:
        doc = self._resolve_doc(request, task, split, doc_id)
        visuals = list(_flatten_visuals(doc_to_visual(doc)))
        if not visuals:
            raise ValueError(
                f"Show-o2 video requires frames or a video path, but {task}/{split}/{doc_id} supplied none"
            )
        if len(visuals) == 1 and isinstance(visuals[0], (str, os.PathLike)):
            path = Path(visuals[0]).expanduser()
            if path.suffix.lower() not in self.VIDEO_SUFFIXES:
                raise ValueError(f"Unsupported video extension for Show-o2: {path.suffix or '<none>'}")
            return self._decode_video(path)

        frames: List[Image.Image] = []
        for visual in visuals:
            if isinstance(visual, Image.Image):
                frames.append(visual.convert("RGB"))
            elif isinstance(visual, (str, os.PathLike)):
                path = Path(visual).expanduser()
                if path.suffix.lower() in self.VIDEO_SUFFIXES:
                    frames.extend(self._decode_video(path))
                else:
                    with Image.open(path) as image:
                        frames.append(image.convert("RGB"))
            else:
                raise TypeError(f"Unsupported video visual type for Show-o2: {type(visual).__name__}")
        return self._uniform_sample(frames, self.num_video_frames)

    def _video_transform(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        width, height = image.size
        scale = min(self.resolution / width, self.resolution / height)
        resized_width = min(int(np.ceil(width * scale)), self.resolution)
        resized_height = min(int(np.ceil(height * scale)), self.resolution)
        resized = image.resize((resized_width, resized_height))
        padded = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        padded.paste(
            resized,
            ((self.resolution - resized_width) // 2, (self.resolution - resized_height) // 2),
        )
        tensor = transforms.ToTensor()(padded)
        return transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)(tensor)

    def _generation_options(self, gen_kwargs: dict[str, Any]) -> tuple[int, float, bool, Optional[int]]:
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = float(gen_kwargs.get("temperature", self.temperature))
        do_sample_value = gen_kwargs.get("do_sample")
        do_sample = temperature > 0 if do_sample_value is None else _as_bool(do_sample_value)
        top_k_value = gen_kwargs.get("top_k", self.top_k)
        top_k = None if top_k_value is None else int(top_k_value)
        if temperature < 0 or (do_sample and temperature == 0):
            raise ValueError("temperature must be positive when sampling and non-negative for greedy decoding")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        if top_k == 0:
            top_k = None
        return max_new_tokens, temperature, do_sample, top_k

    @torch.inference_mode()
    def _answer_video(self, context: str, frames: List[Image.Image], gen_kwargs: dict[str, Any]) -> str:
        self._set_seed()
        context = context.replace("<video>", "").replace("<image>", "").strip()
        max_new_tokens, temperature, do_sample, top_k = self._generation_options(gen_kwargs)
        video = torch.stack([self._video_transform(frame) for frame in frames], dim=0).to(
            self._device, non_blocking=True
        )
        image_latents = self.vae_model.sample(video.unsqueeze(2)).squeeze(2).to(self.weight_type)
        _, _, latent_height, latent_width = image_latents.shape
        patch_size = int(self.config.model.showo.patch_size)
        grid_height, grid_width = latent_height // patch_size, latent_width // patch_size
        image_embeds_und = self._model.image_embedder_und(image_latents)
        image_embeds_gen = self._model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + self._model.position_embedding(self._model.image_position_ids)
        image_embeds_und = self._model.und_trans(image_embeds_und)["last_hidden_state"]
        image_embeds = self._model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))
        expected_per_frame = grid_height * grid_width
        if image_embeds.shape[1] != expected_per_frame:
            raise RuntimeError(
                f"Unexpected Show-o2 frame token grid: got {image_embeds.shape[1]}, expected {expected_per_frame}"
            )
        hidden_size = image_embeds.size(-1)
        image_embeds = image_embeds.reshape(
            len(frames), grid_height, grid_width, hidden_size
        ).permute(0, 3, 1, 2).contiguous()
        image_embeds = F.interpolate(
            image_embeds,
            size=(self.video_token_size, self.video_token_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embeds = image_embeds.permute(0, 2, 3, 1).reshape(1, -1, hidden_size)
        video_token_count = int(image_embeds.shape[1])

        input_ids = self._tokenizer(context, add_special_tokens=False).input_ids
        prefix_ids = [self.showo_token_ids["bos_id"], *self.sys_prompt_ids, *self.role_user_ids]
        suffix_ids = [
            self.showo_token_ids["boi_id"],
            self.showo_token_ids["eoi_id"],
            *input_ids,
            *self.role_assistant_ids,
        ]
        text_tokens_a = torch.tensor(prefix_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        text_tokens_b = torch.tensor(suffix_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        text_embeds_a = self._model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self._model.showo.model.embed_tokens(text_tokens_b)
        if bool(self.config.model.showo.add_time_embeds):
            time_value = torch.tensor([[1.0]], device=self._device)
            time_embeds = self._model.time_embed(time_value, text_embeds_a.dtype)
            if hasattr(self._model, "time_embed_proj"):
                time_embeds = self._model.time_embed_proj(time_embeds)
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
                dim=1,
            ).to(self.weight_type)
            video_offset = text_tokens_a.shape[1] + 2
        else:
            input_embeds = torch.cat(
                [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]],
                dim=1,
            ).to(self.weight_type)
            video_offset = text_tokens_a.shape[1] + 1
        modality_positions = torch.tensor(
            [[[video_offset, video_token_count]]], dtype=torch.long, device=self._device
        )
        attention_mask = self.omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=self._device,
            inverted=True,
        ).to(input_embeds.dtype)
        tokens = self._model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=top_k,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            eos_token=self._tokenizer.eos_token_id,
        )
        if not tokens:
            return ""
        output_ids = torch.stack(tokens).reshape(1, -1)
        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses: List[str] = []
        progress_task = requests[0].task_name if requests else ""
        _report_progress(0, len(requests), progress_task)
        for request in requests:
            try:
                context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            except ValueError as exc:
                raise ValueError(
                    "Show-o2 video expects lmms-eval generate_until arguments "
                    "(context, gen_kwargs, doc_to_visual, doc_id, task, split)"
                ) from exc
            gen_kwargs = dict(gen_kwargs or {})
            frames = self._resolve_video(request, doc_to_visual, doc_id, task, split)
            response = (
                self.dry_run_response
                if self.dry_run
                else self._answer_video(str(context), frames, gen_kwargs)
            )
            response = self._apply_until(response, gen_kwargs)
            responses.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            _report_progress(len(responses), len(requests), task)
        _report_progress(len(responses), len(requests), progress_task, group_finished=True)
        return responses

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Show-o2 video adapter currently supports single-round understanding only")
