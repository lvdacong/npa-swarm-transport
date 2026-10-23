# NPA-Swarm-Transport 本机闭环报告

日期：2026-10-23（Asia/Shanghai）。起点：`cc83926b08ce6ae8e603b56e37a48b6edc63eb40`。

**工程闭环：通过。RL 完整搜索搬运：未通过。**

剩余研究接续：[Issue #2：招募、消息留存与完整搬运](https://github.com/lvdacong/npa-swarm-transport/issues/2)。

最新冻结 search 在 100/101/102 三个开发布局中，均发现初始未知的重物和目的地，但 **0/3 搬运成功、3 完整超时**。没有把 contact、短窗或发现率计作完整任务成功。暂不追加训练种子或消费 1000–1002 保留布局。

## 本机实际验证

- 独立仓库：`C:\Users\ASUS\research-repos\npa-swarm-transport`；独立 `.venv`。没有依赖旧 Swarm，没有修改原 NPA 蜥蜴或停止 AUV Viewer/网络守护进程。
- Windows 11，Ryzen 7 H 260，32 GiB RAM，RTX 5060 Laptop（8151 MiB），驱动 596.13。Python 3.12.10，Torch 2.14.1+cu130，CUDA 13.0，FP32、TF32关闭。
- 最终本机 **45 passed，0 skipped**，含 **4 个实际 CUDA 用例**。Sparse CSR 的 beta 提示保留为 1 条 warning。旧云端/专项存在重叠，不相加；没有重跑 Linux CI。
- CUDA 图、策略输出/梯度和恢复通过对照；生产训练由 OOM 前的 u28 完整状态恢复至 u100。恢复注明源码变更，不声称长轨迹逐位相同。
- 当前原生 Python 源码摘要：`1ccebfdabf5b0c9ee336de78ea6a4b0b31211fafb8f12ef9413ffd8a97f61a5d`。最终 Git 提交及远端读回由 Issue #1 收尾评论绑定。训练、评价、当前源码摘要分别保留在原始 manifest/checkpoint 中。

## 4096 粒子性能与 OOM 修复

起初在近均匀初态，默认32步 rollout / 8步 TBPTT / 3 epochs 的1/2/4环境训练中位吞吐约15/32/62 world-step/s，选4环境；对应推理约37.7/64.3/139.5 world-step/s。

正式搜索后密度增加到每世界约27–30万有向边，原 gather/scatter 在第29次更新 OOM；u28原件及完整堆栈保留。CUDA 消息均值改为 canonical CSR SpMM，保留全部邻居；打乱和重复边也保持原均值与梯度语义。

| 同一密集快照，4×4096，默认PPO | 稳定 world-step/s | 分配显存峰值 GB | 保留池 GB |
| --- | ---: | ---: | ---: |
| 原 gather/scatter 参考 | 47.44 | 2.30 | 4.23 |
| Canonical CSR | 70.40 | 1.94 | 2.50 |

这里 GB=10^9 bytes。吞吐提高约48.4%，保留池下降约41.0%。两次更新的后一次用于稳定值；不是跨场景拼接。

生产 CSR 连续完成72次新增更新至累计u100，训练吞吐中位数约83.3 world-step/s，最大保留池约2.58 GB，没有再次OOM。大缓存清理门槛本段未触发，不能把改善归功于它。

`profile-final` 是8步/4步TBPTT/1epoch的有界训练 trace，包含图、连通、局部GRU、8物理子步、PPO反传和同步。profile数值含采集开销；嵌套范围的百分比不能相加。完整默认配置性能来自另外的非profile测量。

## 学习过程与完整评价分母

所有评价均冻结检查点，保存逐控制步粒子/物体位置、动作及指标。除注明均值动作外，后续采用有记录动作seed的抽样策略。不同阶段重复使用100–102作为开发场景，不能当作独立泛化样本相加。

| 冻结评价 | 成功/请求 | timeout | censored | 异常/中断 | 未开始 | 范围 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 未训练 contact | 0/1 | 1 | 0 | 0 | 0 | 诊断/预热 |
| 原奖励 contact u20 | 0/1 | 1 | 0 | 0 | 0 | 诊断/预热 |
| 力信用 v1 / 均值动作 | 0/3 | 3 | 0 | 0 | 0 | 诊断/预热 |
| 力信用 v1 / 抽样动作 | 0/1 | 1 | 0 | 0 | 0 | 诊断/预热 |
| 扩大感知 v3 | 0/2 | 2 | 0 | 0 | 0 | 诊断/预热 |
| 局部向量 v4 / 感知1.8 | 3/3 | 0 | 0 | 0 | 0 | 诊断/预热 |
| 局部向量 v5 / 感知0.8 | 3/3 | 0 | 0 | 0 | 0 | 诊断/预热 |
| 最终向量结构 contact / 感知0.8 | 3/3 | 0 | 0 | 0 | 0 | 诊断/预热 |
| 正式 search v6 | 0/3 | 3 | 0 | 0 | 0 | 正式4096/1200步 |
| 正式 search v7 | 0/3 | 3 | 0 | 0 | 0 | 正式4096/1200步 |

主要改变及原因：

1. 原奖励下物体不动。增加可选的真实有符号接触冲量信用，先让超过静摩擦门槛的同向合力有可学习信号；仅有局部可见重物的粒子分配距离进展，发现/成功仍共享。
2. 仅调权重仍不能稳定泛化。加入可学习局部向量系数的actor，先用感知1.8预热，再恢复0.8；实际冻结均获得3/3 contact入区。
3. v6正式搜索出现同向漂移到墙角。v7取消无条件世界方向输出分支，仅学习已有局部速度、邻域相对量、目标相对量和接触法向的系数；新覆盖奖励0.01→0.1。没有给网络绝对坐标、全局目标方向或地图，没有角色或动作保护器。

最终结构的 contact 初始检查点继承 v4 u22 → v5 u3 的权重，自己的新段计数为0，并非随机网络；已修正UI展示继承身份。它在100/101/102分别用 106, 114, 97 步成功。contact时限400，仅作课程；search恢复1200及全部原物理尺度。

v7 正式 search 保持4096、感知0.8、未知两端和1200步，三布局均发现两端；最多同时接触数分别为2、3、2，净力最大约1.29、1.54、1.00，均低于静摩擦4；**物体位移三个场景均为0**。因此问题已从发现推进到招募和保持合力，不能称学会完整任务。

contact和两轮正式search的全程采样连通均为0/3。v7两端被发现后的桥保持率为1317/2405≈54.76%；其分母只含两端已发现后的控制步。连通只在控制步端点采样，不是连续时间证明。

训练seed始终是42；场景seed100–102与它不同。尚无正式成功，因此没有做独立训练seed复现；1000–1002没有被请求或查看。保留这些边界，避免把同一策略多布局当成多训练种子。

## 实际浏览器

本机Chrome实际 adapter 为 `amd · rdna-3`（Radeon780M）；原生策略/物理是RTX CUDA。中文、播放/暂停、重置、缩放、粒子#2073选择、Canvas↔WebGPU及1200步终局均有实际操作和像素证据。抽样推理单独使用自身RNG，避免消耗其他代码的全局RNG。

首段未训练诊断完成三完整回合约12分钟播放，DOM保持124，JS堆约35.6–44.7MB，没有单调增长；GPU pass约0.07ms。暂时关闭并恢复自动化默认focus emulation，实际触发document.hidden：隐藏时页面停在1144而原生继续，恢复后追上1200终局。

训练contact检查点另外在硬件页面于92步显示“搬运完成”，明确标记CONTACT；不是正式search成绩。最终search检查点的10分钟持续运行状态为 **PASS**；原始API驱动日志与独立浏览器硬件记录分别保存。若已PASS，终局按原1200步正常重置后继续，结束时暂停。状态率、绘制提交率和GPU计时各自报告，不称显示器FPS；未强制制造物理device loss。

## 检查点、原件与继续入口

较重要的检查点（完整SHA见 `checkpoints.json`）：

- 可重复contact：`runs/local/contact-local-v5/best-distance.pt`，SHA `4989bd05dfc88305dc28a7b451efecfe884da8dde3d62b9dd2614b1a9d8a0b68`。
- 最终向量结构contact：`runs/local/vector-only-init/last.pt`，新段0更新，继承上项；对应 `vector-only-contact-eval` 3/3。
- OOM恢复后的v6：`runs/local/search-csr-v6/last.pt`，SHA `a4a019f8a8a2f8a3b38bf92d4efb52b966c93b62492330d1a8be858bbfe99f4d`。
- 最终search v7：`runs/local/search-vector-only-v7/last.pt`，SHA `64341906974c0fa3689cf408f33e54f0278422b28b74877070b1b1aa6986b18e`。

本机原件根目录是 `C:\Users\ASUS\research-repos\npa-swarm-transport\runs\local`。本报告内manifest为原件副本；其NPZ相对路径应在对应原件目录解析，Git不收纳大型NPZ或PT。证据索引给出全部SHA；交付证据包收纳当前contact和两轮正式search的完整轨迹，较早失败原件仍留在本机并已索引。

可见最终页面在 `http://127.0.0.1:8765/?renderer=webgpu`，最终策略标记为已训练但未证实完整搬运，结束后暂停。若服务已关闭，可在仓库用：

```powershell
.venv\Scripts\python.exe -m npa_swarm serve --checkpoint runs/local/search-vector-only-v7/last.pt --device cuda:0 --stochastic --port 8765
```

剩余研究只针对“已发现两端后，如何通过学习的局部记忆/消息维持招募和净力”，不再重复环境安装、Linux CI或已闭合的CUDA/浏览器验收。需要先获得一条正式完整成功轨迹，再进行必要的独立训练seed及保留布局评价。当前桥/接触证据不能单独证明具体消息机制的因果缺陷，后续应使用局部消息消融和招募课程定位。
