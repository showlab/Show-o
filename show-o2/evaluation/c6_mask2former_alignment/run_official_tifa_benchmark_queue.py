#!/usr/bin/env python3
"""Benchmark official-TIFA self-cycle checkpoints for one adapter route."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Show-o/show-o2')
WORKSPACE = Path('/root/autodl-tmp')
GENEVAL_PYTHON = WORKSPACE / 'venvs/geneval-score/bin/python'
SHOWO_PYTHON = WORKSPACE / 'venvs/showo2-eval/bin/python'
VQA_PYTHON = WORKSPACE / 'venvs/showo2-score-vqa/bin/python'
POSITION_MANIFEST = WORKSPACE / 'data/showo2_layout/position_unseen72_v1.jsonl'
VQA_MODEL = WORKSPACE / 'models/c7/Qwen2.5-VL-7B-Instruct'
TIFA_TEST = WORKSPACE / 'data/showo2_self_cycle/tifa_v1/official_test_unicycle.jsonl'
STEPS = (100, 200, 500, 1000, 1500, 2000, 3865)
VARIANTS = ('s1_m', 's1_mf')


def now(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text()) if Path(p).is_file() else {}
def atomic(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(x, indent=2, sort_keys=True) + '\n'); os.replace(tmp, p)
def parse_steps(v):
    s = tuple(int(x) for x in v.split(',') if x)
    if s != tuple(sorted(set(s))) or not set(s).issubset(STEPS): raise ValueError(v)
    return s

def parse_variants(v):
    variants = tuple(x.strip() for x in v.split(',') if x.strip())
    if not variants or len(set(variants)) != len(variants) or not set(variants).issubset(VARIANTS):
        raise ValueError(v)
    return variants

def run(cmd, log, env=None):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    e = os.environ.copy(); e.update({'HF_HOME': str(WORKSPACE/'cache/huggingface'), 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false', 'CUDA_VISIBLE_DEVICES':'0', 'PYTHONPATH': str(ROOT) + ':' + e.get('PYTHONPATH','')})
    if env: e.update(env)
    with log.open('a') as f:
        f.write(f'[{now()}] command={cmd!r}\n'); f.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0: raise RuntimeError(f'command failed ({result.returncode}): {cmd!r}')

def checkpoint(root, variant, route, step):
    label = f'{variant}_{route}'
    p = Path(root) / label / 'checkpoints' / f'step-{step:06d}.pt'
    if not p.is_file(): raise FileNotFoundError(p)
    return p

def variant_name(variant, route, step): return f'{variant}_{route}_step_{step:06d}'
def geneval_root(run_root, variant, route, step): return Path(run_root)/'geneval'/variant/route/f'step-{step:06d}'
def vqa_root(run_root, route, step): return Path(run_root)/'vqascore'/route/f'step-{step:06d}'
def vsr_root(run_root, variant, route, step): return Path(run_root)/'vsr'/variant/route/f'step-{step:06d}'
def tifa_root(run_root, variant, route, step): return Path(run_root)/'tifa_test'/variant/route/f'step-{step:06d}'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--training-root', type=Path, required=True)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--route', choices=('lora','g_adapter'), required=True)
    ap.add_argument('--steps', default='100,200,500,1000,2000,3865')
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--eval-seed', type=int, default=20260810)
    args=ap.parse_args(); steps=parse_steps(args.steps); variants=parse_variants(args.variants)
    rr=args.run_root.resolve(); rr.mkdir(parents=True, exist_ok=True)
    labels={v: f'{v}_{args.route}' for v in variants}
    tasks=['preflight'] + [f'geneval__{v}__{s}' for s in steps for v in variants] + [f'vqa__{s}' for s in steps] + [f'vsr__{v}__{s}' for s in steps for v in variants] + [f'tifa__{v}__{s}' for s in steps for v in variants] + ['aggregate']
    status_path=rr/'queue_status.json'; started=time.time()
    def done(task):
        if task=='preflight': return load(rr/'preflight.json').get('status')=='completed'
        if task=='aggregate': return (rr/'BENCHMARK_QUEUE_COMPLETED').is_file()
        k,*p=task.split('__')
        if k=='geneval':
            root=geneval_root(rr,p[0],args.route,int(p[1])); return (root/'BENCHMARK_COMPLETED').is_file()
        if k=='vqa': return load(vqa_root(rr,args.route,int(p[0]))/'summary.json').get('status')=='completed'
        if k=='vsr': return load(vsr_root(rr,p[0],args.route,int(p[1]))/'summary.json').get('status')=='completed'
        if k=='tifa': return load(tifa_root(rr,p[0],args.route,int(p[1]))/'summary.json').get('status')=='completed'
        return False
    def write(phase='running', current=None, error=None):
        completed=[t for t in tasks if done(t)]
        current_eta=None
        if len(completed)>0 and len(completed)<len(tasks): current_eta=(len(tasks)-len(completed))*((time.time()-started)/len(completed))
        atomic(status_path, {'status': 'failed' if error else ('completed' if len(completed)==len(tasks) else 'running'), 'phase': phase, 'current': current, 'total_tasks': len(tasks), 'completed_tasks': len(completed), 'progress': 100*len(completed)/len(tasks), 'eta_seconds': current_eta, 'tasks': [{'id':t,'status':'completed' if t in completed else ('running' if t==current else 'waiting')} for t in tasks], 'route':args.route, 'steps':list(steps), 'training_root':str(args.training_root.resolve()), 'tifa_test':str(TIFA_TEST.resolve()), 'updated_at':now(), 'error':error})
    try:
        for required in (GENEVAL_PYTHON, SHOWO_PYTHON, VQA_PYTHON, POSITION_MANIFEST, VQA_MODEL/'model.safetensors.index.json', TIFA_TEST, ROOT/'Wan2.1_VAE.pth'):
            if not Path(required).exists(): raise FileNotFoundError(required)
        for v in variants:
            route_dir=Path(args.training_root)/labels[v]
            st=load(route_dir/'status.json')
            if st.get('status')!='completed' or int(st.get('step',-1)) != 3865: raise RuntimeError(f'incomplete training: {route_dir}')
            for s in steps: checkpoint(args.training_root,v,args.route,s)
        atomic(rr/'preflight.json', {'status':'completed','route':args.route,'steps':list(steps),'variants':labels,'tifa_test':str(TIFA_TEST.resolve()),'completed_at':now()})
        write('benchmarking')
        # Geneval, VQAScore, VSR and official TIFA test are intentionally separate tasks.
        for s in steps:
            for v in variants:
                task=f'geneval__{v}__{s}'; root=geneval_root(rr,v,args.route,s)
                if not done(task):
                    write('geneval',task); ck=checkpoint(args.training_root,v,args.route,s); name=variant_name(v,args.route,s); cfg=root/'variants.json'; cfg.parent.mkdir(parents=True,exist_ok=True); atomic(cfg,{name:{'guidance_steps':0,'inner_steps':1,'energy_step_size':1.0,'layers':[7],'attention_source':'qwen','max_grad_rms':0.0,'drsa_checkpoint':str(ck.resolve())}})
                    run([str(SHOWO_PYTHON),'-u',str(ROOT/'evaluation/c3_layout_guidance/run_quick8_v2.py'),'--run-dir',str(root),'--protocol-version',f'official-tifa-{args.route}-geneval-{v}-{s}','--manifest',str(POSITION_MANIFEST),'--config',str(ROOT/'configs/showo2_1.5b_demo_432x432.yaml'),'--model','showlab/show-o2-1.5B','--vae-path',str(ROOT/'Wan2.1_VAE.pth'),'--variant-config-json',str(cfg),'--variants',name,'--samples-per-prompt','4','--sample-batch-size','4','--base-seed','12026','--guidance-scale','5.0','--num-inference-steps','12','--guided-guidance-steps','0','--inside-weight','1.0','--cross-weight','0.0','--relation-weight','0.0','--limit-prompts','72','--geneval-python',str(GENEVAL_PYTHON),'--geneval-root',str(WORKSPACE/'geneval'),'--geneval-model-config',str(WORKSPACE/'mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),'--geneval-model-path',str(WORKSPACE/'models/geneval'),'--geneval-clip-path',str(WORKSPACE/'models/geneval/clip/ViT-L-14.state_dict.pt')],root/'run.log')
                    (root/'BENCHMARK_COMPLETED').touch()
            task=f'vqa__{s}'; out=vqa_root(rr,args.route,s)
            if not done(task):
                write('vqascore',task); cmd=[str(VQA_PYTHON),'-u',str(ROOT/'evaluation/c7_generation_benchmark/score_vqa.py')]
                for v in variants:
                    name=variant_name(v,args.route,s); source=geneval_root(rr,v,args.route,s)/name/'geneval'; cmd += ['--variant',f'{v}={source}']
                cmd += ['--output-dir',str(out),'--model-path',str(VQA_MODEL),'--batch-size','8','--attn-implementation','sdpa']; run(cmd,out/'run.log')
        for s in steps:
            for v in variants:
                task=f'vsr__{v}__{s}'; out=vsr_root(rr,v,args.route,s)
                if not done(task):
                    write('vsr',task); run([str(GENEVAL_PYTHON),'-u',str(ROOT/'evaluation/c6_mask2former_alignment/eval_g_adapter_vsr.py'),'--run-dir',str(out),'--checkpoint',str(checkpoint(args.training_root,v,args.route,s)),'--split','dev','--bridge','on'],out/'run.log')
        for s in steps:
            for v in variants:
                task=f'tifa__{v}__{s}'; out=tifa_root(rr,v,args.route,s)
                if not done(task):
                    write('tifa_test',task); run([str(SHOWO_PYTHON),'-u',str(ROOT/'evaluation/c8_self_cycle/run_self_cycle.py'),'--run-dir',str(out),'--dataset',str(TIFA_TEST),'--config',str(ROOT/'configs/showo2_1.5b_demo_432x432.yaml'),'--model','showlab/show-o2-1.5B','--variant-name',variant_name(v,args.route,s),'--drsa-checkpoint',str(checkpoint(args.training_root,v,args.route,s)),'--vae-path',str(ROOT/'Wan2.1_VAE.pth'),'--guidance-scale','5.0','--num-inference-steps','50','--generation-batch-size','4','--samples-per-prompt','1','--base-seed',str(args.eval_seed),'--max-open-answer-tokens','12'],out/'run.log')
        atomic(rr/'comparison_summary.json',{'status':'completed','protocol':'official-tifa-test-v1','route':args.route,'steps':list(steps),'variants':list(variants),'training_root':str(args.training_root.resolve()),'tifa_test':str(TIFA_TEST.resolve()),'results_root':str(rr.resolve()),'completed_at':now()})
        (rr/'BENCHMARK_QUEUE_COMPLETED').touch(); write('completed'); print(json.dumps(load(status_path),indent=2)); return 0
    except Exception as e:
        write('failed',error=f'{type(e).__name__}: {e}'); (rr/'BENCHMARK_QUEUE_FAILED').touch(); traceback.print_exc(); return 1
if __name__=='__main__': raise SystemExit(main())
