"""Only run these on real CUDA. Cloud CPU reports skips, not passes."""
import copy
import math
import pytest
import torch
from npa_swarm.config import EnvConfig,TrainConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.graph import Graph,aggregate,radius_graph
from npa_swarm.policy import LocalPolicy
from npa_swarm.train import Trainer

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="requires a physical CUDA device")

@pytest.fixture(autouse=True)
def fp32_reference():
    a,b=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    yield
    torch.backends.cuda.matmul.allow_tf32=a;torch.backends.cudnn.allow_tf32=b


def test_cuda_radius_graph_and_policy_backward_match_cpu():
    gen=torch.Generator().manual_seed(61);pos=torch.rand(2,64,2,generator=gen);pos[0,1]=pos[0,0]
    cg,gg=radius_graph(pos,.29),radius_graph(pos.cuda(),.29)
    assert torch.equal((cg.src*128+cg.dst).sort().values,(gg.src*128+gg.dst).sort().values.cpu())
    torch.manual_seed(12);cp=LocalPolicy(20,16);gp=copy.deepcopy(cp).cuda()
    ob=torch.randn(2,64,20,generator=gen);h=torch.randn(2,64,16,generator=gen)
    a,b=cp(ob,h,cg),gp(ob.cuda(),h.cuda(),gg)
    for k in (0,2,3):torch.testing.assert_close(b[k].cpu(),a[k],atol=2e-5,rtol=2e-4)
    sum(a[k].square().mean() for k in (0,2,3)).backward()
    sum(b[k].square().mean() for k in (0,2,3)).backward()
    for (name,x),(_,y) in zip(cp.named_parameters(),gp.named_parameters()):
        if x.grad is not None:
            assert torch.isfinite(y.grad).all(),name
            torch.testing.assert_close(y.grad.cpu(),x.grad,atol=2e-5,rtol=2e-4,msg=name)


def test_cuda_state_restore_continues_with_tolerance():
    cfg=EnvConfig(n_agents=48,num_envs=2,stage="contact",episode_steps=20);env=SwarmEnv(cfg,"cuda")
    gen=torch.Generator().manual_seed(83)
    actions=[torch.rand(2,48,2,generator=gen).cuda()*2-1 for _ in range(5)]
    for a in actions[:2]:env.step(a)
    restored=SwarmEnv(cfg,"cuda");restored.load_state_dict(env.state_dict())
    for action in actions[2:]:
        a,b=env.step(action),restored.step(action)
        for k in (0,1):torch.testing.assert_close(b[k],a[k],atol=5e-5,rtol=5e-4)
        for k in (2,3):assert torch.equal(a[k],b[k])
    torch.testing.assert_close(restored.positions,env.positions,atol=5e-5,rtol=5e-4)
    assert torch.equal(restored.steps,env.steps) and torch.equal(restored.generator.get_state(),env.generator.get_state())


def test_cuda_ppo_checkpoint_next_update():
    # Use pytest temporary directory to preserve isolation.
    import tempfile
    from pathlib import Path
    cfg=TrainConfig(hidden_dim=16,rollout_steps=4,bptt_steps=2,ppo_epochs=1)
    t=Trainer(SwarmEnv(EnvConfig(n_agents=32,stage="contact",episode_steps=3),"cuda"),cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"c.pt";t.update();t.save(path)
        em=t.update();weights={k:v.detach().cpu().clone() for k,v in t.policy.state_dict().items()};h=t.hidden.detach().cpu().clone()
        r=Trainer.load(path,"cuda");am=r.update()
        assert r.updates==2 and r.env_steps==8 and r.episodes==t.episodes
        assert all(math.isfinite(v) for v in am.values() if isinstance(v,float))
        for key,v in r.policy.state_dict().items():torch.testing.assert_close(v.cpu(),weights[key],atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(r.hidden.cpu(),h,atol=1e-4,rtol=1e-3)
        assert abs(am["mean_reward"]-em["mean_reward"])<1e-4


def test_cuda_sparse_mean_duplicates_values_gradients():
    g=Graph(torch.tensor([2,0,1,0]),torch.tensor([1,1,0,1]),3);gg=g.to("cuda")
    x=torch.tensor([[[1.,2.],[3.,4.],[7.,8.]]],requires_grad=True);y=x.detach().cuda().requires_grad_()
    a,b=aggregate(x,g),aggregate(y,gg);torch.testing.assert_close(b.cpu(),a,atol=2e-6,rtol=2e-5)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(y.grad.cpu(),x.grad,atol=2e-6,rtol=2e-5)
    m=gg._mean_matrix
    torch.sparse_csr_tensor(m.crow_indices(),m.col_indices(),m.values(),size=m.shape,check_invariants=True)
