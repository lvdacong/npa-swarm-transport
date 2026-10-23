# NPA Swarm Transport

900 粒子、局部消息与可学习记忆、循环 PPO、真实接触合力搬运。独立项目；不修改原蜥蜴，不引入旧 Swarm 模块。

**当前是任务 v2 的工程候选，不是已学会搬运的策略。** v1 的 contact 3/3 与 search 0/3 结果仍保留；不能把旧结果算到新物理、规模和模型上。

## 使用

在已有兼容 CUDA 的环境内安装本项目，不必重装已通过的驱动或 Torch：

```powershell
python -m pip install -e ".[test]"
python -m npa_swarm probe --device cuda:0 --agents 900 --envs 1 --steps 16 --train-updates 1 --output runs/v2/probe
python -m npa_swarm train --device cuda:0 --envs 2 --updates 20 --output runs/v2/search
python -m npa_swarm serve --device cuda:0 --checkpoint runs/v2/search/last.pt --stochastic
```

浏览器打开终端显示的 localhost 地址。只使用 WebGPU；没有 Canvas 或渲染器菜单。播放、重置、倍速、缩放与选中粒子保留；诊断默认折叠。`∞` 表示取消人为等待，不承诺硬件能达到某个倍速。模型未训练时明确标示，不能当成功演示。

首个 20 更新只是观察窗，不是收敛期限，不要求机械跑满。不使用 contact→search 或轻物→重物课程。

## v2 改动

默认 900 粒子，半径 0.06，通信半径 0.48，感知半径 1.2，场地仍为 20×20。单体最大自由速度从 0.5 调到 1.0。货物采用连续阻力：单个粒子能够慢推，多粒子同向接触更快；不再有默认静摩擦死区。动作不受连通保护，断连不删除粒子。奖励包含真实距离进展、实际移动功信用、脱群成本及有符号重连进展；没有固定角色或手工招募控制器。

两端通路要求同一个**粒子连通分量**分别进入两端表面外的通信半径；“看见两端”不算通路，物体不能充当免费中继。通路与全群连通分开统计。

网络不输入绝对坐标、ID、地图或全局目标方向。两个通用、可学习的方向记忆槽可以随一跳消息传递；不预设哪个槽代表目的地。v1 权重不兼容，不静默转换或继续旧优化器计数。

[任务定义](docs/TASK.md) · [结构](docs/ARCHITECTURE.md) · [本机接续](docs/LOCAL_HANDOFF.md) · [本轮验收与边界](reports/cloud-revision/README.md)

## 验证原则

```powershell
python -m pytest -q
python scripts/browser_smoke.py --hardware --headed --executable "C:\Program Files\Google\Chrome\Application\chrome.exe" --output reports/local-v2/browser
```

日常只跑改动相关检查。没有自动 push/PR CI，也没有 Linux CI；可选 Windows 工作流仅手动触发，本轮未运行。真实 GPU、实际浏览器、学习质量依赖本机证据。历史 v1 性能辅助脚本可能只适用于 `8e07995`；本轮入口为上述 CLI 与更新后的 browser_smoke，不需要复跑旧验收矩阵。
