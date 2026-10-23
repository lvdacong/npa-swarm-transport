import argparse
import json
import sys
import torch
from .config import EnvConfig, TrainConfig
from .diagnostics import probe, resolve_device, system_info, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local-message swarm transport, task v2")
    sub = parser.add_subparsers(dest="command", required=True)
    def device_args(p):
        p.add_argument("--device",default="auto")
        p.add_argument("--threads",type=int,default=2)
    def env_args(p):
        p.add_argument("--agents",type=int,default=900)
        p.add_argument("--envs",type=int,default=1)
        p.add_argument("--stage",choices=("contact","search"),default="search",help="contact is diagnostic only; no curriculum required")
        p.add_argument("--layout",choices=("random","diagonal"),default="random")
        p.add_argument("--seed",type=int,default=42)
        p.add_argument("--episode-steps",type=int,default=1200)
        p.add_argument("--sense-radius",type=float,default=1.2)
        p.add_argument("--work-credit-weight",type=float,default=1.,help="actual moving-work credit; population-mean experiment uses N")
        device_args(p)
    doctor=sub.add_parser("doctor"); device_args(doctor);doctor.add_argument("--output")
    perf=sub.add_parser("probe");env_args(perf)
    perf.add_argument("--output",required=True);perf.add_argument("--steps",type=int,default=16)
    perf.add_argument("--warmup",type=int,default=2);perf.add_argument("--train-updates",type=int,default=0)
    perf.add_argument("--trace",action="store_true")
    training=sub.add_parser("train");env_args(training)
    training.add_argument("--output",required=True);training.add_argument("--updates",type=int,required=True)
    training.add_argument("--resume");training.add_argument("--warmstart",help="v2 weights only; not a v1 checkpoint conversion")
    training.add_argument("--rollout-steps",type=int,default=128);training.add_argument("--bptt-steps",type=int,default=32)
    training.add_argument("--ppo-epochs",type=int,default=3);training.add_argument("--hidden-dim",type=int,default=32)
    training.add_argument("--learning-rate",type=float,default=3e-4)
    evaluation=sub.add_parser("evaluate");device_args(evaluation)
    evaluation.add_argument("--checkpoint",required=True);evaluation.add_argument("--output",required=True)
    evaluation.add_argument("--seeds",nargs="+",type=int,default=[100,101,102]);evaluation.add_argument("--episodes-per-seed",type=int,default=1)
    evaluation.add_argument("--stage",choices=("contact","search"),default="search")
    evaluation.add_argument("--layout",choices=("random","diagonal"))
    evaluation.add_argument("--max-steps",type=int,help="short diagnostic, not a full evaluation")
    evaluation.add_argument("--stochastic",action="store_true")
    viewer=sub.add_parser("serve");env_args(viewer)
    viewer.add_argument("--checkpoint");viewer.add_argument("--untrained",action="store_true")
    viewer.add_argument("--stochastic",action="store_true");viewer.add_argument("--port",type=int,default=8765)
    viewer.add_argument("--host",default="127.0.0.1")
    args=parser.parse_args(argv)
    if args.threads<1:parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=resolve_device(args.device)
    if args.command=="doctor":
        result=system_info(str(device))
        if args.output:write_json(args.output,result)
    elif args.command=="evaluate":
        from .evaluate import evaluate
        result=evaluate(args.checkpoint,args.output,args.seeds,args.episodes_per_seed,device=str(device),
                        stage=args.stage,max_steps=args.max_steps,deterministic=not args.stochastic,layout=args.layout)
    else:
        cfg=EnvConfig(n_agents=args.agents,num_envs=args.envs,stage=args.stage,layout=args.layout,
                      seed=args.seed,episode_steps=args.episode_steps,sense_radius=args.sense_radius,
                      work_credit_weight=args.work_credit_weight)
        if args.command=="probe":
            result=probe(cfg,args.output,str(device),args.steps,args.warmup,args.train_updates,args.trace)
        elif args.command=="train":
            from .train import train
            if args.resume and args.warmstart:parser.error("Choose --resume OR --warmstart")
            if args.resume:
                forbidden={"--agents","--envs","--stage","--layout","--seed","--episode-steps","--sense-radius","--work-credit-weight",
                           "--rollout-steps","--bptt-steps","--ppo-epochs","--hidden-dim","--learning-rate"}
                provided=sys.argv[1:] if argv is None else argv
                if any(token.split("=",1)[0] in forbidden for token in provided):
                    parser.error("Strict resume uses saved configs; omit overrides or start a new run")
            tc=TrainConfig(hidden_dim=args.hidden_dim,rollout_steps=args.rollout_steps,bptt_steps=args.bptt_steps,
                           ppo_epochs=args.ppo_epochs,learning_rate=args.learning_rate,seed=args.seed)
            result=train(None if args.resume else cfg,None if args.resume else tc,args.output,args.updates,
                         str(device),resume=args.resume,warmstart=args.warmstart)
        else:
            from .server import serve
            serve(config=cfg,checkpoint=args.checkpoint,untrained=args.untrained,device=str(device),
                  port=args.port,host=args.host,stochastic=args.stochastic)
            return
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if args.command=="evaluate" and not result.get("engineering_evaluation_completed",True):
        raise SystemExit(1)
