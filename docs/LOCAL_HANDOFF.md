# Codex 本机接续：完成真实 GPU 与学习闭环

本机接续已执行，当前证据和可直接加载的检查点见 [2026-10-23本机报告](../reports/local-20261023/README.md)。安装、CUDA、性能、恢复和浏览器不必机械重跑。研究剩余问题是：未知两端均能发现后，尚不能招募足够粒子形成搬运合力。contact成功不是正式search成功；后续范围和起点由主Issue收尾链接给出。

用户已授权 Codex 在本项目内自主修改、优化、训练、评价和修复，不需要逐步申请确认。云端已提供完整首版代码和执行入口；本机任务是用实际反馈把它做成，不是机械复跑清单。

## 目标与边界

独立新仓库 `lvdacong/npa-swarm-transport`。不迁入旧 Swarm 模块；原 NPA 蜥蜴保持原状。正式目标约4096粒子、局部感知与消息记忆、NN不输入绝对坐标、真实接触合力搬运、无连通保护器。首版采用断连软惩罚且继续回合。可以根据证据调整实现、奖励、课程和资源设置，变化在报告中说明即可。

先读当前主Issue、`README.md`、`docs/TASK.md`、`reports/cloud-20261023/README.md`。Issue中的发布提交是起点；若本机已有后续工作，先核对差异再合流，不覆盖现场。无需重新验收蜥蜴、MNIST或Lego。

## 1. 独立安装与现场采集

在自己的项目目录克隆仓库。以下为 Windows PowerShell，目录名与端口均可自主调整。先读取本机当前GPU负载，安排短测与已有任务的使用时段；正式吞吐与延迟测量在明确设备归属和负载条件下进行。

```powershell
git clone https://github.com/lvdacong/npa-swarm-transport.git
Set-Location npa-swarm-transport
git status --short
git rev-parse HEAD
py -3.12 -m venv .venv
$SwarmPython = Join-Path $PWD '.venv\Scripts\python.exe'
nvidia-smi
# 已核对存在的官方 Windows Python 3.12 CUDA wheel。按实际驱动选合适版本；不要求为此全局重装环境。
& $SwarmPython -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
& $SwarmPython -m pip install -e '.[test]'
& $SwarmPython -m npa_swarm doctor --device cuda:0 --output runs/local/system.json
& $SwarmPython -m pytest -q --junitxml=runs/local/pytest.xml
```

若驱动不支持该CUDA构建，按[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)使用与现有驱动/GPU兼容的版本。报告实际Python、Torch、CUDA构建、驱动、GPU型号、显存与测试结果。`cuda:0` 是CUDA编号，不能据此推断Windows任务管理器编号或浏览器选择了哪块卡。

测试包含3个真实CUDA用例；本机应该实际运行它们。云端的跳过不是GPU通过。CPU已验证完整状态恢复；CUDA的浮点累加顺序可能不同，区分状态恢复正确与长轨迹逐位相同。

## 2. 有界性能与资源探针

```powershell
& $SwarmPython -m npa_swarm probe --agents 4096 --envs 1 --stage search --device cuda:0 --steps 16 --warmup 2 --train-updates 1 --trace --output runs/local/probe4096
```

这个入口包含真实图构建、局部GRU、8子步接触物理、连通计算和一个8步PPO更新。保存每项耗时、峰值显存和trace，**不会把未训练动作记作学习效果**。`probe`的8步/1epoch配置是资源探针，正式默认是32步rollout/8步TBPTT/3epoch。

若trace/显存说明有余量，再比较2个或4个并行环境，选更好的总world-step吞吐。保持粒子规模和任务语义；低显存占用不等于应该增加粒子或随意改变batch。若GPU被小算子、动态邻域长度或同步拖慢，可直接优化实现并做一个小输入数值对照；不要先展开多个新后端。FP32是可比较起点，不预设AMP/TF32一定更好。

关注：模型forward/反传、图分桶/连通、物理子步、主机同步各占多少；`n_agents`、并行环境、rollout长度、实际显存和world/agent-step吞吐都要明确。图不截邻居，极端聚团边数仍可能很大。

## 3. 先让有效合力出现，再做完整搜索

```powershell
& $SwarmPython -m npa_swarm train --stage contact --agents 4096 --envs 1 --device cuda:0 --updates 50 --output runs/local/contact
& $SwarmPython -m npa_swarm evaluate --checkpoint runs/local/contact/last.pt --stage contact --seeds 100 101 102 --episodes-per-seed 1 --device cuda:0 --output runs/local/contact-eval
# 需要继续时；updates表示新增次数。
& $SwarmPython -m npa_swarm train --resume runs/local/contact/last.pt --device cuda:0 --updates 50 --output runs/local/contact
```

50次是首段观察窗，不是收敛期限，也不是必须盲目消耗的预算。看是否产生接触、**有效同向净推力、可重复物体位移和距离下降**，并与未训练策略作一个必要对照。如果只有抱团、回报升高但物体不动，直接检查奖励量级、接触动力学和信用分配，修改后复验。无需先填满种子或继续没有证据的长训。

有可重复的预热进展后，按同样规模转入完整 `search`：

```powershell
& $SwarmPython -m npa_swarm train --stage search --agents 4096 --envs 1 --warmstart runs/local/contact/last.pt --device cuda:0 --updates 50 --output runs/local/search
& $SwarmPython -m npa_swarm evaluate --checkpoint runs/local/search/last.pt --stage search --seeds 100 101 102 --episodes-per-seed 3 --device cuda:0 --output runs/local/search-eval
```

`warmstart`只迁移权重，重置优化器、环境、记忆和计数，记录父检查点；不是原轨迹精确续训。`resume`恢复已保存配置、状态和累计计数，命令不要混入被忽略的配置覆盖。

先取得一条真实成功路线，再补独立训练种子的复验。评价layout seeds与训练seed是不同概念，不能把同一策略在9个场景的评价称为3训练种子复现。开发评价用100–102；最后留出的场景用1000–1002，事先保留到候选冻结后。任何需要根据holdout改动的情况，明确标成新的开发迭代。

完整任务必须保持未知两端初态、真实物理、完整时限和正式规模。小N或缩短`--max-steps`仅用于诊断，评价器会标明。保留全部请求、失败、超时、删失、异常和未开始分母；评价输出含冻结检查点、每步原始轨迹及摘要。

## 4. 真实浏览器与设备确认

```powershell
& $SwarmPython -m playwright install chromium
& $SwarmPython scripts/browser_smoke.py --particles 4096 --renderer canvas --output runs/local/browser
& $SwarmPython scripts/browser_smoke.py --particles 4096 --renderer webgpu --output runs/local/browser
& $SwarmPython -m npa_swarm serve --checkpoint runs/local/search/last.pt --device cuda:0 --port 8765
```

自动化浏览器脚本用于回归，软件adapter结果不替代真实硬件验收。用本机可见Chrome/Edge打开 `http://127.0.0.1:8765`，验证播放/暂停、重置、缩放、粒子选择、中文、模型身份与终局状态。记录浏览器adapter（可能是AMD核显）、原生模拟GPU、渲染模式、唯一状态率、绘制提交率、GPU timer、实际操作体验；不得把rAF/插值帧当作新模拟状态或显示器呈现FPS。

可在训练同时用核显显示，但要确认实际设备，并记录共享CPU、功耗、散热影响。用于验收的性能测量请说明并发负载。无硬件隔离保证；本项目不引入上一项目的复杂租约系统。

完成一次适度持续的可见运行（建议10分钟，出现问题就针对性修复），看资源是否持续增长、控制是否积压、页面隐藏/恢复与device-loss回退是否正确。首轮不需要泛化到多个前端后端或大量浏览器组合。

## 5. 回传与结束条件

```powershell
& $SwarmPython scripts/collect_evidence.py --root runs/local --output reports/local-evidence --archive
```

该工具只生成文件摘要和证据包，不自动判研究成功。检查点记录SHA-256，较大的原件可保留在本机并说明位置；代码、摘要、必要日志/图表/失败轨迹链接在同一个主Issue回填。

回传重点：

1. 实际提交/源码摘要、改动原因、硬件/软件、关键测试通过/失败。
2. 4096下真实训练配置、world/agent吞吐、峰值显存、瓶颈及选择依据。
3. 发现重物/目的地、形成两端通路、有效推动、最终入区的完整轨迹；搬运成功、采样全程连通和桥保持率各自统计。
4. 成功与失败的冻结评价分母；独立训练种子与holdout结果，或明确尚未完成。
5. 物理浏览器adapter、真实操作及持续运行结果；保留软件/硬件区别。

连通当前仅检查控制步末。若要声称每个物理子步都连通，先实现并运行对应测量；不需要为了通过而加动作保护器。

工程闭环通过与任务学会搬运分别报告；不要把最终研究字段固定为true来凑结果。没有主要问题就给出完成结论，不为了次要问题循环发布新Issue。存在重大未解问题时，在旧Issue交代证据与当前状态，再创建范围明确的新接续Issue。
