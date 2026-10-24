# Issue #5 本机闭环（2026-10-25）

**工程闭环通过；RL完整搬运未通过。** 本轮证明了真实900粒子轨迹上的后侧奖励可以进入GAE和actor梯度，但没有学会持续接触与正确推动。旧900 CUDA / WebGPU / 4.001×播放证据沿用local-v3，未重跑浏览器、4096或CI。本轮25–26 world steps/s是包含PPO优化的训练吞吐，不是播放倍速。

## 修复与证据

从 `d183b43d751b0197785eaf18493bc659306e98d2` 开始，父A20校验为 `46e2761cd7ba95117eda47d04658c78caa670e9feb5ff1498462cd7a818bf3a1`。现场无其他训练；网络进程及旧暂停Viewer未改动，原NPA/旧Swarm未使用。

1. 旧检查点缺relay字段时恢复原.97，避免新默认.997偷偷改变旧策略。缺rear字段恢复0，缺potential版本恢复1，缺eligibility恢复main。新实验均独立目录显式warmstart。
2. 原后侧势在半径1.8、cos0→.2时gap .465667→.426，错误奖励+.031733。首次v2修正又把前侧角向信号变成平台（失败保留）。v3在完整感知环上向前转不获正奖、前侧径向不获奖、转向后侧有奖。CPU/CUDA定向单调性与物理不变测试通过。
3. main门槛真实冻结384步后活跃0、rear奖励/优势/actor梯度贡献均0，虽然cargo附近已有目标消息。新增可选 `--rear-eligibility neighbors`，允许有实际邻居的小分量，孤立者排除；**这是显式改变训练奖励资格**，默认仍main。全群连通代价、桥、成功标准不变。
4. neighbors冻结同轨迹：8194活跃粒子步，rear绝对奖励和240.418686，按原main反事实仅.000925255；条件reward均值+.004182/绝对均值.029341；GAE贡献norm35.82275，actor head梯度变化norm.00182557（full norm.02325514）。见 `connected-learning-audit.json`。这是无优化的梯度审计，不是学会任务；独立CUDA轨迹不承诺位级相同。
5. 真实更新仍接触流失且方向反复。新增可选 `--local-batch-fraction .5`：全体PPO目标与活跃样本条件优势目标各半，critic对应加权；全轨迹归一权重，保持时间顺序/BPTT。默认0，不推荐把失败实验设默认。无新actor输入、脚本动作或角色。空集/单例回退全体目标。单例回退在最后训练之后修复，CPU/CUDA增量通过，**未声称重训后的结果**。

## 实际训练（所有world steps，未计冻结审计）

| 本机 runs/v4 子目录 | 设置 | 更新 | world steps | 完整训练回合/成功 | 停止原因 |
|---|---|---:|---:|---:|---|
| rear-a20 | 原势v1/main，128步窗 | 1 | 256 | 0/0 | 活跃0，先查门槛/几何 |
| rear-fixed-a20 | 失败势v2/main，128步窗 | 10 | 2560 | 2/0 | 仅55活跃粒子步，奖励全0 |
| rear-angular-a20 | v3/main，384步窗 | 0 | 0 | 0/0 | 冻结审计信号0，未训练 |
| rear-connected-a20 | v3/neighbors，384步窗 | 3 | 2304 | 0/0 | 接触衰减，净力再次反向 |
| rear-balanced | 从connected-u3权重warmstart，局部权重.5 | 2 | 1536 | 0/0 | 活跃198→1，接触消失 |

累计16更新、6656 world steps、5,990,400 agent steps，完整训练回合2、成功0。其余是保留的部分回合，不混入完整评价分母。所有训练都是900×2、未知两端search、重物原物理、1200步、relay .997、rear .8、work1、lr3e-4、32-BPTT/3epoch；没有课程。完整配置、父链、检查点SHA和精确源码SHA见 `run-manifest.json`。逐更新指标见 `*-metrics.json`。

connected u1/u2/u3距离进展为-.014965/+.004365/-.042074，朝目标力-.004179/+.002199/-.016648，接触比例.1315/.1081/.09245。活跃3933/3777/2008粒子步，活跃critic MSE1.55/1.54/1.69，其他.697/.445/.331。不是单纯消息没有到达，而是局部动作信用和持续接触未学成。

connected每更新768 world steps耗时29.22–29.56秒，26.0–26.3 world steps/s，峰值allocated最高717,204,480B。balanced耗时30.45–30.63秒，25.1–25.2 world steps/s，峰值1,053,515,776B。它们不是严格同状态性能A/B，不推断因果倍速。

## 冻结评价与分母

| 候选 | 请求/完成 | 成功/超时 | 异常/删失/中断/未启动 | 距离进展 | 最长接触 | 严格桥 |
|---|---|---|---|---:|---:|---|
| rear-fixed-u10 seed100 | 1/1 | 0/1 | 0/0/0/0 | -.006668 | .2秒 | 0/855 |
| rear-balanced-u2 seed100 | 1/1 | 0/1 | 0/0/0/0 | 0 | 0秒 | 0/1020 |

两次均完整1200步、发现两端、全程连通0/1；它们是**相同开发seed、不同候选**，不是独立布局成功率0/2。没有扩100–102/对角103，也未开启保留布局或独立训练seed。最终候选实际物体位移0、接触0，虽cargo附近目标消息覆盖.70588、方向误差约6.2e-7，仍没有搬运。

最终失败定位：后侧奖励只有保持可见且已有消息/邻接的样本才可用，粒子离开cargo后其学习信号消失。重加权能改变更新，但不能凭空产生持续接触样本；局部critic此前对活跃者明显过高估计。本轮没有证明局部加权足够，也没有证明所有RL方法不可能。下一次应先检查同轨迹上消息对局部动作的因果影响、接触退出时的时间信用与局部价值估计；不要继续同配方长训或盲增奖励。

## 复现与保留

仓库根为 `C:/Users/ASUS/research-repos/npa-swarm-transport`。大权重/NPZ不入Git，保留本机：
- `runs/v4/*/{initial,last,update-XXXX}.pt`（实际文件清单/SHA见manifest；不是所有run都有每更新文件）。
- `runs/v4/source-rear*.zip` 保存各阶段精确源码；最终源文件由本次提交保存。
- `reports/local-v4/{rear10-dev100,balanced-dev100}/episode-0000.npz`、checkpoint.pt、manifest、summary及trajectory-analysis。
- 原A20与本轮失败权重未覆盖。无后台训练/新增排程，旧Viewer仍展示旧模型，并非本轮候选。

可复查冻结审计：`.venv/Scripts/python.exe scripts/audit_rear_learning.py runs/v4/rear-connected-a20/initial.pt reports/local-v4/new-audit.json`。CUDA非位级重复，因果消融在同一次轨迹内计算。

增量测试证据为本目录XML：奖励→GAE→actor（CPU/真CUDA）、旧配置恢复、CUDA PPO恢复、几何方向、neighbors物理/验收不变、局部采样空集/单例回退及参数更新。未重复旧软件矩阵或CI。局部采样初始测试未固定熵估计RNG而失败，修正为共同RNG后通过；不把随机估计差异当算法问题。
