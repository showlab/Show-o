
import logging
from typing import Dict, List, Optional
import torch
from collections import defaultdict


logger = logging.getLogger(__name__)

STATS_TYPES = ['max_abs', 'min_abs', 'var']


class Stats:
    def __init__(self, device: str = "cpu"):
        self._lists: Dict[str, List[torch.Tensor]] = {name: [] for name in STATS_TYPES}
        self.device = device

    def append(self, name: str, tensor: torch.Tensor) -> None:
        if name not in self._lists:
            raise KeyError(f"Unknown stat name '{name}'")
        self._lists[name].append(tensor.detach().to('cpu'))

    def append_many(self, stats: Dict[str, torch.Tensor]) -> None:
        for k, v in stats.items():
            if v is None:
                continue
            self.append(k, v)

    @torch.no_grad()
    def collect_from_tensor(self, out: torch.Tensor) -> None:
        max_abs = out.abs().amax(dim=1, keepdim=False)           # (batch_size,)
        var = out.var(dim=1, unbiased=False, keepdim=False)      # (batch_size,)
        min_abs = out.abs().amin(dim=1, keepdim=False)   
        
        self.append("max_abs", max_abs)
        self.append("var", var)
        self.append("min_abs", min_abs)

    def cat(self, name: str) -> Optional[torch.Tensor]:
        lst = self._lists.get(name)
        if not lst:
            return None
        return torch.cat(lst, dim=0)
    
    def clear(self) -> None:
        for k in list(self._lists.keys()):
            self._lists[k].clear()

    def __repr__(self):
        s = ", ".join(f"{k}: {len(v)}" for k, v in self._lists.items())
        return f"Stats({s})"
        
def get_activations(recorder, stat: str = "max_abs", layers: list[int] = None):
    if stat not in STATS_TYPES:
        raise ValueError(f"stat must be one of {STATS_TYPES}, got {stat}")
    if layers is None:
        layers = list(range(24))
    layer_names = []
    acts_list = []

    for ind in layers:
        layer_name = f"showo.model.layers.{ind}.mlp.fc2"
        if layer_name not in recorder.outputs:
            continue

        stats_obj: Stats = recorder.outputs[layer_name]
        t = stats_obj.cat(stat)
        if t is None:
            continue

        t = t.detach().cpu().flatten()
        if t.numel() == 0:
            continue

        layer_names.append(layer_name)
        acts_list.append(t)

    return layer_names, acts_list


class LayerOutputRecorder:
    def __init__(self, device='cuda'):
        self.outputs = defaultdict(lambda: Stats(device=device))
        self.inputs_shapes = defaultdict(list)
        self.handles = []
        self.device = device

    def build_hook_fn(self, name):
        def hook_fn(module, input_, output):
            with torch.no_grad():
                out = output.detach()
                self.outputs[name].collect_from_tensor(out)
                self.inputs_shapes[name].append(input_[0].shape)
        return hook_fn

    def register_hook(self, module_name, module):
        handle = module.register_forward_hook(self.build_hook_fn(module_name))
        self.handles.append(handle)

    def register_hooks(self, modules: list[tuple[str, torch.nn.Module]]) -> None:
        for module_name, module in modules:
            self.register_hook(module_name, module)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def clear(self):
        for stats in self.outputs.values():
            stats.clear()
        self.outputs.clear()
        self.inputs_shapes.clear()
        torch.cuda.empty_cache()