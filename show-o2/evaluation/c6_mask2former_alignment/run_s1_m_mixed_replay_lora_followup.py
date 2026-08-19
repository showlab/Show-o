import argparse,json,os,subprocess,time,traceback
from pathlib import Path
W=Path('/root/autodl-tmp'); R=Path('/root/autodl-tmp/Show-o/show-o2')
MF=W/'runs/showo2_tifa_mixed_replay/official-tifa-mixed-s1-lora-20260819T000000Z/mf_benchmark_step1500'
LOCK=W/'data/showo2_self_cycle/tifa_v1/official_tifa_lora_starts.lock.json'
MAN=W/'data/showo2_self_cycle/tifa_v1/official_trainval_interleaved.manifest.json'
REPLAY=W/'data/drsa/mask_logits_coco_multiaxis_v1/train.jsonl'
SHARDS=W/'data/drsa/spright_coco/data'
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def load(p):
 try:return json.loads(Path(p).read_text())
 except:return {}
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2)+'\n'); os.replace(t,p)
def main():
 a=argparse.ArgumentParser(); a.add_argument('--run-root',type=Path,required=True); a.add_argument('--max-steps',type=int,default=3865); a.add_argument('--save-steps',default='100,200,300,400,500,750,1000,1250,1500,2000,2500,3865'); a.add_argument('--benchmark-steps',default='100,200,500,1000,1500,2000,3865'); a.add_argument('--seed',type=int,default=20260820); a.add_argument('--eval-seed',type=int,default=20260810); a.add_argument('--learning-rate',type=float,default=1e-5); a.add_argument('--replay-probability',type=float,default=.25); a.add_argument('--replay-mask-weight',type=float,default=1.); a.add_argument('--replay-flow-weight',type=float,default=.1); a.add_argument('--teacher-consistency-weight',type=float,default=.25); a.add_argument('--teacher-temperature',type=float,default=1.); a.add_argument('--train-python',type=Path,default=W/'venvs/geneval-score/bin/python'); a.add_argument('--eval-python',type=Path,default=W/'venvs/showo2-eval/bin/python'); x=a.parse_args(); root=x.run_root.resolve(); root.mkdir(parents=True,exist_ok=True); sp=root/'queue_status.json'
 d={'status':'waiting','stage':'waiting_for_mf_followup','current':None,'wait_for':str(MF),'route':'s1_m_lora','max_steps':x.max_steps,'save_steps':[int(z) for z in x.save_steps.split(',')],'benchmark_steps':[int(z) for z in x.benchmark_steps.split(',')],'updated_at':now()}
 def status(stage,cur=None,st='running',detail=None,error=None): d.update(status=st,stage=stage,current=cur,current_detail=detail or {},error=error,updated_at=now()); put(sp,d)
 e=os.environ.copy(); e.update(OMP_NUM_THREADS='1',PYTHONPATH=str(W/'mmdetection-2.28.2')+':'+str(R),HF_HOME=str(W/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0')
 def run(cmd,log,child,stage,cur):
  log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('a') as f:
   f.write(f'[{now()}] command={cmd!r}\n'); f.flush(); p=subprocess.Popen(cmd,cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT)
   while p.poll() is None: status(stage,cur,detail={'pid':p.pid,'child_status_path':str(child),**load(child)}); time.sleep(10)
  if p.returncode: raise RuntimeError(f'{cur} failed rc={p.returncode}')
 try:
  lock=load(LOCK); start=Path(next(z for z in lock['starts'] if z['branch']=='S1-M')['path']); man=load(MAN); d['init_checkpoint']=str(start); status('waiting_for_mf_followup')
  while not (MF/'BENCHMARK_QUEUE_COMPLETED').is_file():
   if (MF/'FOLLOWUP_FAILED').is_file(): raise RuntimeError('MF followup failed')
   status('waiting_for_mf_followup',detail=load(MF/'queue_status.json')); time.sleep(30)
  smoke=root/'preflight_smoke'
  if not (smoke/'TRAINING_COMPLETED').is_file():
   c=[str(x.train_python),'-u',str(R/'evaluation/c6_mask2former_alignment/train_tifa12_self_cycle.py'),'--run-dir',str(smoke),'--manifest',man['train'],'--shard-root',man['shard_root'],'--init-checkpoint',str(start),'--label','s1_m_mixed_replay_smoke','--mode','smoke','--max-steps','1','--save-steps','1','--seed',str(x.seed),'--learning-rate',str(x.learning_rate),'--status-every','1','--log-every','1','--allow-unpaired','--generation-start','pure_noise','--min-valid-questions','1','--replay-manifest',str(REPLAY),'--replay-shard-root',str(SHARDS),'--replay-probability','1.0','--replay-mask-weight',str(x.replay_mask_weight),'--replay-flow-weight',str(x.replay_flow_weight),'--teacher-consistency-weight',str(x.teacher_consistency_weight),'--teacher-temperature',str(x.teacher_temperature)]
   run(c,smoke/'run.log',smoke/'status.json','preflight_smoke','s1_m_step_1')
  if not (smoke/'TRAINING_COMPLETED').is_file(): raise RuntimeError('S1-M smoke failed')
  train=root/'s1_m_lora'; train.mkdir(parents=True,exist_ok=True)
  if not (train/'TRAINING_COMPLETED').is_file():
   c=[str(x.train_python),'-u',str(R/'evaluation/c6_mask2former_alignment/train_tifa12_self_cycle.py'),'--run-dir',str(train),'--manifest',man['train'],'--shard-root',man['shard_root'],'--init-checkpoint',str(start),'--label','s1_m_lora','--max-steps',str(x.max_steps),'--save-steps',x.save_steps,'--seed',str(x.seed),'--learning-rate',str(x.learning_rate),'--status-every','1','--log-every','1','--allow-unpaired','--generation-start','pure_noise','--min-valid-questions','1','--replay-manifest',str(REPLAY),'--replay-shard-root',str(SHARDS),'--replay-probability',str(x.replay_probability),'--replay-mask-weight',str(x.replay_mask_weight),'--replay-flow-weight',str(x.replay_flow_weight),'--teacher-consistency-weight',str(x.teacher_consistency_weight),'--teacher-temperature',str(x.teacher_temperature)]
   run(c,train/'run.log',train/'status.json','training','s1_m_lora')
  if not (train/'TRAINING_COMPLETED').is_file(): raise RuntimeError('S1-M training failed')
  b=root/'benchmark'
  if not (b/'BENCHMARK_QUEUE_COMPLETED').is_file():
   c=[str(x.eval_python),'-u',str(R/'evaluation/c6_mask2former_alignment/run_official_tifa_benchmark_queue.py'),'--training-root',str(root),'--run-root',str(b),'--route','lora','--variants','s1_m','--steps',x.benchmark_steps,'--eval-seed',str(x.eval_seed)]
   run(c,b/'run.log',b/'queue_status.json','benchmark','s1_m_lora')
  if not (b/'BENCHMARK_QUEUE_COMPLETED').is_file(): raise RuntimeError('S1-M benchmark failed')
  status('completed',st='completed'); (root/'QUEUE_COMPLETED').touch(); return 0
 except Exception as ex: status('failed',d.get('current'),st='failed',error=f'{type(ex).__name__}: {ex}'); (root/'QUEUE_FAILED').touch(); traceback.print_exc(); return 1
if __name__=='__main__': raise SystemExit(main())
