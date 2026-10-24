import copy
import json
import numpy as np
import pytest
import torch
from npa_swarm.config import EnvConfig,TrainConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.graph import Graph,radius_graph
from npa_swarm.policy import LocalPolicy,squashed_log_prob
from npa_swarm.train import Trainer,compute_gae,load_policy_checkpoint,train,capture_rng,restore_rng
from npa_swarm.evaluate import evaluate


def tiny(episode_steps=5):
    return Trainer(SwarmEnv(EnvConfig(n_agents=16,stage="contact",episode_steps=episode_steps)),
                   TrainConfig(hidden_dim=8,rollout_steps=4,bptt_steps=2,ppo_epochs=2))


def test_squashed_density_and_extremes():
    mean=torch.tensor([[[.2,-.4]]]);ls=torch.tensor([-.5,.1]);z=torch.tensor([[[.7,-1.3]]])
    dist=torch.distributions.TransformedDistribution(torch.distributions.Normal(mean,ls.exp()),
        [torch.distributions.transforms.TanhTransform(cache_size=1)])
    torch.testing.assert_close(squashed_log_prob(mean,ls,z),dist.log_prob(z.tanh()).sum(-1))
    assert torch.isfinite(squashed_log_prob(mean,ls,z*100)).all()


def test_timeout_bootstraps_terminal_not_next_episode():
    r=torch.tensor([1.,2.,3.]).view(3,1,1);v=torch.zeros_like(r)
    next_v=torch.tensor([10.,20.,30.]).view(3,1,1)
    a,ret=compute_gae(r,v,next_v,torch.tensor([[False],[False],[True]]),
        torch.tensor([[True],[False],[False]]),.5,1.)
    torch.testing.assert_close(a.flatten(),torch.tensor([6.,13.5,3.]));torch.testing.assert_close(ret,a)


def test_messages_one_hop_and_gradient():
    torch.manual_seed(7);p=LocalPolicy(20,8)
    g=Graph(torch.tensor([0,1,1,2]),torch.tensor([1,0,2,1]),3)
    a=torch.zeros(1,3,20);b=a.clone();b[0,0,0]=1;b.requires_grad_()
    ha,hb=p.initial_state(1,3),p.initial_state(1,3)
    for t in range(3):
        *_,ha=p(a,ha,g);*_,hb=p(b,hb,g)
        if t==0:torch.testing.assert_close(ha[:,1:],hb[:,1:],atol=0,rtol=0)
        if t==1:torch.testing.assert_close(ha[:,2:],hb[:,2:],atol=0,rtol=0)
    assert not torch.allclose(ha[:,2],hb[:,2],atol=1e-7)
    hb[:,2].square().sum().backward();assert b.grad[0,0].abs().sum()>0


def test_permutation_equivariance():
    torch.manual_seed(11);x=torch.rand(2,12,2);p=LocalPolicy(20,8)
    obs,h=torch.randn(2,12,20),torch.randn(2,12,p.state_dim);order=torch.randperm(12)
    a=p(obs,h,radius_graph(x,.6));b=p(obs[:,order],h[:,order],radius_graph(x[:,order],.6))
    for k in (0,2,3):torch.testing.assert_close(b[k],a[k][:,order],atol=1e-6,rtol=1e-5)


def test_direction_memory_survives_absence_of_local_cue_without_global_bias():
    p=LocalPolicy(20,8);g=Graph(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long),1)
    obs=torch.zeros(1,1,20);obs[...,6]=.4;obs[...,16:]=1
    h=p.initial_state(1,1);h[...,:p.hidden_dim-4]=torch.randn(1,1,p.hidden_dim-4)
    with torch.no_grad():
        p.actor.weight.zero_();p.actor.bias.fill_(1)
        p.vector_write.weight.zero_();p.vector_write.bias.zero_()
    mean,_,_,_=p(obs,h,g);assert torch.count_nonzero(mean)==0
    h[...,p.hidden_dim-4:p.hidden_dim-2]=torch.tensor([.7,.2])
    mean,_,_,carry=p(obs,h,g)
    assert mean.norm()>0 and carry[...,p.hidden_dim-4:p.hidden_dim].norm()>0
    # This is architectural capability, not proof that training learns a destination.


def test_recurrent_ppo_updates_action_memory_and_messages():
    t=tiny();before=copy.deepcopy(t.policy.state_dict());m=t.update()
    assert m["update"]==1 and m["env_steps"]==4
    assert np.isfinite([v for v in m.values() if isinstance(v,float)]).all()
    after=t.policy.state_dict()
    for key in ("actor.weight","memory.weight_hh","encode.0.weight","vector_write.weight"):
        assert not torch.equal(before[key],after[key]),key
    assert not torch.equal(before["encode.0.weight"][:,-8:],after["encode.0.weight"][:,-8:])


@pytest.mark.parametrize("horizon",[3,100])
def test_bootstrap_reuse_matches_eager_reference_including_resets(horizon):
    t=tiny(episode_steps=horizon);env_state=t.env.state_dict();h=t.hidden.clone();rng=capture_rng()
    optimized=t.collect()
    env=SwarmEnv(t.env.cfg);env.load_state_dict(env_state);restore_rng(rng,"cpu")
    rewards=[];values=[];next_values=[];terms=[];truncs=[];latent=[]
    with torch.no_grad():
        for _ in range(t.cfg.rollout_steps):
            a,z,lp,v,m=t.policy.act(env.observe(),h,env.graph)
            ob,r,term,trunc,_=env.step(a)
            _,_,nv,_=t.policy(ob,m,env.graph)
            rewards.append(r);values.append(v);next_values.append(nv);terms.append(term);truncs.append(trunc);latent.append(z)
            done=term|trunc;h=m*(~done).view(-1,1,1)
            if bool(done.any()):env.reset(done)
    adv,ret=compute_gae(torch.stack(rewards),torch.stack(values),torch.stack(next_values),
                        torch.stack(terms),torch.stack(truncs),t.cfg.gamma,t.cfg.gae_lambda)
    for a,b in ((optimized.latent_actions,torch.stack(latent)),(optimized.advantages,adv),(optimized.returns,ret)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_resume_next_update_across_reset_is_exact_cpu(tmp_path):
    t=tiny(3);t.update();path=tmp_path/"resume.pt";t.save(path)
    expected=t.update();weights=copy.deepcopy(t.policy.state_dict());world=t.env.state_dict();h=t.hidden.clone()
    loaded=Trainer.load(path);actual=loaded.update()
    assert loaded.updates==2 and loaded.env_steps==8 and loaded.episodes==t.episodes>0
    for key,v in weights.items():torch.testing.assert_close(loaded.policy.state_dict()[key],v,rtol=0,atol=0)
    for key,v in world.items():
        if torch.is_tensor(v):torch.testing.assert_close(loaded.env.state_dict()[key],v,rtol=0,atol=0)
    torch.testing.assert_close(loaded.hidden,h,rtol=0,atol=0)
    assert actual["mean_reward"]==expected["mean_reward"] and actual["policy_loss"]==expected["policy_loss"]


def test_budget_resume_and_v3_warmstart(tmp_path):
    ec=EnvConfig(n_agents=16,stage="contact",episode_steps=3)
    tc=TrainConfig(hidden_dim=8,rollout_steps=2,bptt_steps=2,ppo_epochs=1)
    out=tmp_path/"run";first=train(ec,tc,out,1)
    resumed=train(None,None,out,1,resume=out/"last.pt")
    assert first["updates"]==1 and resumed["updates"]==2 and resumed["env_steps"]==4
    with pytest.raises(FileExistsError):train(ec,tc,out,1)
    warm=train(EnvConfig(n_agents=16,episode_steps=3),tc,tmp_path/"warm",0,warmstart=out/"last.pt")
    assert warm["updates"]==0 and warm["lineage"]["parent_updates"]==2
    _,cfg,meta=load_policy_checkpoint(tmp_path/"warm/last.pt")
    assert cfg.stage=="search" and meta["has_trained_weights"] and not meta["learned_success_verified"]


def test_old_schema_rejected_and_failed_load_keeps_rng(tmp_path):
    path=tmp_path/"old.pt";torch.save({"schema_version":1},path)
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="v1|schema|Schema|version"):Trainer.load(path)
    assert torch.equal(before,torch.get_rng_state())


def test_evaluation_preserves_timeout_and_trajectory(tmp_path):
    t=tiny(2);path=tmp_path/"policy.pt";t.save(path);out=tmp_path/"eval"
    s=evaluate(path,out,seeds=(17,18),episodes_per_seed=1)
    assert s["requested"]==s["completed"]==s["timeout"]==2 and s["diagnostic"]
    assert not s["research_task_passed"]
    manifest=json.loads((out/"manifest.json").read_text())
    for r in manifest["requests"]:
        tr=np.load(out/r["trajectory"])
        assert tr["positions"].shape==(3,16,2) and tr["actions"].shape==(2,16,2)
        assert len(json.loads(str(tr["metrics_json"])))==3


def test_evaluation_error_keeps_denominator(tmp_path,monkeypatch):
    t=tiny();path=tmp_path/"p.pt";t.save(path)
    def fail(*a):raise RuntimeError("deliberate failure")
    monkeypatch.setattr(SwarmEnv,"step",fail)
    s=evaluate(path,tmp_path/"err",seeds=(42,43),episodes_per_seed=1)
    assert s["requested"]==2 and s["error"]==1 and s["not_started"]==1
    assert not s["engineering_evaluation_completed"]


def test_short_evaluation_is_censored(tmp_path):
    t=tiny(10);path=tmp_path/"p.pt";t.save(path)
    s=evaluate(path,tmp_path/"short",seeds=(5,),episodes_per_seed=1,max_steps=1)
    assert s["censored"]==1 and s["completed"]==0 and not s["full_horizon_completed"]


def test_sampled_evaluation_repeats_action_seeds(tmp_path):
    t=tiny(2);path=tmp_path/"p.pt";t.save(path)
    for name in ("a","b"):
        evaluate(path,tmp_path/name,seeds=(19,),episodes_per_seed=1,stage="contact",deterministic=False)
    a=np.load(tmp_path/"a/episode-0000.npz");b=np.load(tmp_path/"b/episode-0000.npz")
    np.testing.assert_array_equal(a["actions"],b["actions"]);np.testing.assert_array_equal(a["positions"],b["positions"])


def test_modified_scales_are_explicit_diagnostics(tmp_path):
    t=tiny(2);t.env.cfg.sense_radius=1.8;path=tmp_path/"p.pt";t.save(path)
    s=evaluate(path,tmp_path/"eval",seeds=(19,),episodes_per_seed=1)
    assert s["diagnostic"] and s["task_config_differences"]["sense_radius"]=={"actual":1.8,"reference":1.2}


def test_source_identity_kept_when_files_change(tmp_path,monkeypatch):
    t=tiny();identity=t.runtime_source_sha256
    monkeypatch.setattr("npa_swarm.train.source_sha256",lambda:"changed-on-disk")
    path=tmp_path/"p.pt";t.save(path)
    assert torch.load(path,weights_only=False)["source_sha256"]==identity


def test_relay_retention_is_checkpointed_and_restored(tmp_path):
    t = Trainer(SwarmEnv(EnvConfig(n_agents=16)), TrainConfig(hidden_dim=8, relay_memory_decay=.997))
    path = tmp_path / 'retention.pt'; t.save(path)
    loaded = Trainer.load(path)
    policy, _, metadata = load_policy_checkpoint(path)
    assert loaded.policy.relay_memory_decay == policy.relay_memory_decay == metadata['relay_memory_decay'] == .997
