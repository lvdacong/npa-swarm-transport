# Codex 本机接续：v2 / 900 粒子

用户已授权本仓库自主修改、优化、训练和复验，无须逐步请示。只保护其他项目的现场；不要动蜥蜴，不追加旧 Swarm 或完整 NPA 复现。

## 先读这几项变化

旧 `8e07995` / Issue #2 的 4096、静摩擦 4、contact 预热及 v1 权重要求已被本轮替代。新的起点是 **900、连续阻力、直接完整 search、可学习方向记忆、WebGPU-only**。不是继续 v1 长训；旧工程结果保留，旧 contact 3/3 不计为 v2 成功。v1 检查点不能加载进这个结构；使用新运行目录，不转换旧优化器状态。原日志/权重不删除。

云端没有 CUDA。45 个 CPU 测试通过、4 个 CUDA 测试跳过；真实默认 900/128-rollout/32-BPTT/3-epoch 单次更新跑通。新浏览器导航被云端 Chromium 管理策略拦截；只有静态 HTML/CSS 布局和 JS 语法核验，不是 WebGPU 像素通过。详见 reports/cloud-revision/README.md。本轮没有运行 Actions，工作流已改为手动 Windows 单任务。

## 1. 复用环境，先验改动

先只读检查当前 HEAD、现场修改和其他 GPU 任务；安全获取发布提交后安装 editable 包。原已验证的 Python/Torch/CUDA/驱动能用就复用，不再重装整套环境或跑旧矩阵。

```powershell
python -m pip install -e ".[test]"
python -m npa_swarm doctor --device cuda:0 --output reports/local-v2/system.json
python -m pytest -q tests/test_cuda.py
python -m npa_swarm probe --device cuda:0 --agents 900 --envs 1 --steps 16 --warmup 3 --train-updates 1 --output runs/v2/probe
```

CPU 全回归不必重复 Linux CI。若本机对新代码作了修改，跑对应的定向回归；必要时一次 `python -m pytest -q` 足够。CUDA 浮点容差与逐位恢复分开说明。

probe 的 8 步训练只是软件/资源探针，不代表下面默认 128/32 的显存。先让正式命令完成一个真实更新查看显存/耗时，再决定并行环境数。通常先试 1 或 2 个，确有余量再增加，不铺开完整参数矩阵。真实聚团、离群时的图与奖励开销也要看，不能只测初态。

## 2. 直接训练 search，不做课程

```powershell
python -m npa_swarm train --device cuda:0 --envs 2 --updates 20 --output runs/v2/search-seed42
# 需要继续时，额外更新的数量明确记录，配置从检查点恢复：
python -m npa_swarm train --device cuda:0 --resume runs/v2/search-seed42/last.pt --updates 20 --output runs/v2/search-seed42
```

20 更新只是初始观察窗，不是强制预算或收敛期限。不要因为一次短窗未发现目标就判方法无效，也不要原封不动无期限追加。

先追踪：发现两端的模拟时间、发现后接触人数、持续接触时间、净有效推力、真实物体位移和目标距离减少、脱群比例/持续时间、通路。**奖励变高而物体不动不是进展。** 单推能动已经过物理诊断，但招募、保持接触、消息长距离传播仍是训练假设。向量记忆是可学习能力，不是已训练的导航。

若无进展，按机制定位：探索是否只靠随机抖动；是否尚未碰到物体；是否方向信息未传播/遗忘；还是接触滑脱/推力抵消。不先轮换大量 seeds、奖励权重或后端。不把全球目标方向输入 actor，不加入脚本招募/动作保护/删粒子。必要算法、记忆、奖励、并行参数可自主修改并复验；新运行标明变更，不冒充严格恢复。

有实际位移后再保存冻结候选，检查完整回合；`candidate.pt` 仅按训练中实际距离改善保存，不是通过认证的 best model。

## 3. 一条冻结评价链

```powershell
python -m npa_swarm evaluate --checkpoint runs/v2/search-seed42/last.pt --device cuda:0 --stochastic --seeds 100 101 102 --output reports/local-v2/dev
python scripts/analyze_trajectories.py reports/local-v2/dev
python -m npa_swarm evaluate --checkpoint runs/v2/search-seed42/last.pt --device cuda:0 --stochastic --layout diagonal --seeds 103 --output reports/local-v2/diagonal
```

先证实完整 search 可成功，再做必要独立训练种子复验；布局 seed 不是训练 seed。1000–1002 仅在候选冻结后作为保留场景，不反复用来调参。保留成功/失败/超时/删失/错误/未开始分母，完整动作、粒子、货物轨迹。不得用 contact、小 N、固定推力、缩短回合代替任务成功。

桥接是同一粒子分量接近两个端点的**通信包络**，不是感知包络；端点不作免费中继。不把桥接与全群连通混为一谈。当前只是控制步采样，不能宣称连续时间从未断开。请人工抽查至少一段“桥断开/重新建立”的轨迹与显示，确认新判据符合画面。

## 4. 精简 WebGPU 与速度

```powershell
python scripts/browser_smoke.py --hardware --headed --executable "C:\Program Files\Google\Chrome\Application\chrome.exe" --output reports/local-v2/browser
python -m npa_swarm serve --device cuda:0 --checkpoint runs/v2/search-seed42/last.pt --stochastic
```

Chrome 路径按实际安装调整；Edge 同理。烟测是未训练、CPU 模拟、真实浏览器设备的集成检查。随后用上面 checkpoint 服务实际观看代表性轨迹。没有渲染菜单、Canvas fallback 或大标题，诊断只在 `⋯` 里。页面不可用/设备丢失要报错，不能换 Canvas 后宣称 WebGPU。

先检查新 shader 能编译、900 实例有真实像素、播放/暂停/重置/选择/缩放和终局。云端受策略阻塞的烟测**本机首次必须运行并按实际错误修复**；旧 v1 浏览器 10 分钟成功不能替代它。只需一次代表性持续运行，不要求重跑旧 10 分钟矩阵。

倍速提高的是模拟目标推进率；记录**实际 simulated-seconds/wall-second**，而不只写按钮 4×。区分三项：物理速度、模拟吞吐、显示提交率。若 4× 达不到，先看当前一控制步耗时及阻塞，不先提高物理 dt；若吞吐足但粒子行动慢，检查策略动作/有效接触，不优化 shader 冒充行为改善。

原生 GPU 与浏览器 adapter 分别记录。900 包体 28,928 bytes，发布最多 30Hz；没有端到端零拷贝。保留 CSR，先检查图/组件/奖励距离、policy、物理子步、反传，只有瓶颈明确才做 compile/融合/精度等改变。不得为更高 GPU 利用率牺牲总体吞吐或交互。

## 回填与停止

在新的同一接续 Issue 回传：代码 SHA、简要硬件版本/设备、默认或实际配置、显存和吞吐、第一条完整搬运轨迹、失败边界、评价分母与连通统计、WebGPU 设备及实际倍速。大权重保留本机路径+摘要即可。分开给“工程通过”和“学习通过”。若阶段还没学会，报告实际失败，不用已通过的软件工程结果包装完成。只有重大未解问题才换 Issue，普通修复留在本单，避免循环发布。

## v3 relay experiment — targeted addendum

Start from the published relay revision (not any prior v2 checkpoint). Keep the last v2 failed trajectories/checkpoint as immutable baseline; v3 needs a **fresh direct search run** because the message architecture has changed (schema 3). No contact curriculum, no scripted controller, no action shield. First use the existing CUDA environment and a short two-world smoke/training window; check finite gradients, saved-state resume and actual world-step throughput. Pay special attention to the new two reductions and extra edge-relative vectors: if they cut throughput materially, optimize them before longer training. Do **not** repeat old Windows/Linux CI or the hardware-browser soak unless a relevant code change breaks that path. On actual search trajectories log how many nodes acquire cargo/goal relay confidence, distance progress and signed goal-directed cargo force, bridge after discovery, contact persistence and training reward terms. Only then choose a bounded continuation. Preserve failure denominators and independent seeds; stop/rethink when learning gets worse, rather than boosting reward globally or mechanically training longer. Work and report within the active handoff Issue.
