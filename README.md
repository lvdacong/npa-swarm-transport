# NPA Swarm Transport

**局部感知 → 局部消息与记忆 → RL 动作 → 真实接触合力搬运。**

一个独立、最小的二维群体强化学习项目。以 **4096 粒子**为正式目标规模，借鉴 NPA 的局部重复更新机制，重新实现任务、物理、策略和训练。没有迁入旧 Swarm 模块，也没有改动原 NPA 蜥蜴项目。

> 当前是可执行的研究基线，**尚无“策略已经学会完整搬运”的结论**。云端验收与本机交接见 [交付报告](reports/cloud-20261023/README.md) 和 [Codex 接续文档](docs/LOCAL_HANDOFF.md)。

2026-10-23 本机结果见 [本机报告](reports/local-20261023/README.md)：真实 RTX CUDA、4096 粒子训练/恢复和硬件 Chrome 已验证；最终回归45项通过。局部接触预热冻结评价3/3成功，完整未知两端搜索已能3/3发现两端，搬运仍0/3。原始失败、评价分母和检查点摘要均保留。

## 已实现的主线

- **物理环境：**同质圆盘粒子、一个较重圆盘物体、一个目的地区域。有限推力、接触弹簧、阻力和静摩擦；物体只能被实际接触力推动。
- **局部策略：**20 维局部观测、32 维 GRU 记忆、邻域消息平均聚合、共享二维动作策略。网络不接收绝对坐标、粒子 ID、全局目标方向或全局地图。
- **RL：**顺序重演的 recurrent PPO，时间截断反传；完整环境、记忆、优化器和 RNG 检查点。支持精确状态续接及单独标记的权重预热。
- **连接：**没有连通保护器、自动重连或预设角色。断连只产生软惩罚，回合继续；搬运完成、全群连通、两端通路分别评价。
- **性能：**Torch CPU/CUDA 通用的空间分桶邻域，生产路径不建全粒子 N×N 矩阵，不静默截断邻居。WebGPU 实例化渲染、GPU 双快照插值、二进制传输与 Canvas 回退。

## 安装与第一次运行

Python 3.12。创建本项目独立虚拟环境。CPU 云端验收使用 Torch **2.14.1+cpu**；本机 GPU 安装见 [Windows 步骤](docs/LOCAL_HANDOFF.md)。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
python -m npa_swarm doctor --device cpu
python -m pytest -q
```

运行 **4096 粒子的未训练诊断页面**，打开 `http://127.0.0.1:8765`，点击“开始”：

```bash
python -m npa_swarm serve --untrained --agents 4096 --device cpu
```

页面明确标记未训练策略。它不会悄悄启动训练，也不把随机运动展示为已经学会搬运。使用训练检查点时：

```bash
python -m npa_swarm serve --checkpoint runs/contact/last.pt --device cuda:0
```

## 一条训练路线

先测设备负载，再训练局部可见两端的 `contact` 预热，随后初始化完整未知目标的 `search` 任务。预热和诊断成功都不算完整任务成功。

```bash
python -m npa_swarm probe --agents 4096 --device cuda:0 --steps 16 --train-updates 1 --output runs/probe
python -m npa_swarm train --stage contact --agents 4096 --device cuda:0 --updates 50 --output runs/contact
python -m npa_swarm train --resume runs/contact/last.pt --device cuda:0 --updates 50 --output runs/contact
# 仅在预热确有有效推力与进展后进入 search；warmstart 重置优化器、环境和计数并保留父检查点身份。
python -m npa_swarm train --stage search --warmstart runs/contact/last.pt --agents 4096 --device cuda:0 --updates 50 --output runs/search
python -m npa_swarm evaluate --checkpoint runs/search/last.pt --seeds 100 101 102 --episodes-per-seed 3 --device cuda:0 --output runs/search-eval
```

`--updates` 始终表示**新增 PPO 更新数**，恢复不会重置累计预算。50 次更新是观察反馈的起始短段，**不是收敛承诺或必须机械跑满的预算**。完整实验应根据发现率、接触推力、物体进展和冻结评价决定下一步。

## 需要知道的边界

`search` 初始两端都在所有粒子的感知范围之外；粒子初始连通。观测中的相对向量使用公共模拟坐标轴，因此这里只承诺**无绝对坐标输入**，没有额外宣称旋转等变或无共同方向参考。

连通性在每个控制步末（默认 0.1 秒）检查。报告中的“全程连通”指所有被检查时刻，**不是连续时间或全部物理子步的证明**。初期任务成功按整个物体进入目的地判定，连通性另行报告。

空间分桶减少普通密度下的搜索成本，极端聚团仍可能产生大量真实邻居。CPU/CUDA 都保留实际邻域，具体 CUDA 性能需本机测量。

原生模拟状态传到浏览器仍有设备→主机→网络→浏览器 GPU 的必要复制；前端复用 GPU 缓冲并避免重复状态上传，**不宣称整条链路零拷贝**。页面绘制提交率也不等于显示器实际呈现 FPS。

## 文档

- [任务与奖励定义](docs/TASK.md)
- [结构与数据流](docs/ARCHITECTURE.md)
- [本机接续与验收](docs/LOCAL_HANDOFF.md)
- [云端实际证据](reports/cloud-20261023/README.md)

架构启发：[Neural Particle Automata 官方实现](https://github.com/TheDevilWillBeBee/NPA)。本项目为重新实现，不使用其预训练权重，也不将原蜥蜴能力当作新 RL 任务能力。
