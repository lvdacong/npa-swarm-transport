# 云端首版交付与实际验收

## 结论

独立项目的环境、局部消息策略、循环 PPO、完整恢复、冻结评价、中文浏览器和本机命令均已实现。正式目标保持 **4096 粒子**。没有迁入旧 Swarm 模块，没有改动原 NPA 蜥蜴仓库。

**工程代码可接续；完整搜索与合作搬运尚未学会或验收。** 云端没有 CUDA，也没有物理浏览器 GPU/显示器。Canvas 和显式软件 WebGPU 均已实际通过完整浏览器检查；早期设备丢失与空呈现失败保留，并经独立最小复现定位到 Chromium 软件后端启动配置。远端 CI 的最终结果及发布提交由主接续 Issue 记录，不能用本文件预先推断。

## 实际结果

| 检查 | 实际结果 | 原始证据 |
| --- | --- | --- |
| 最终 Python 全回归 | **32 passed、3 skipped**；3 个跳过均需要 CUDA | [pytest.txt](pytest.txt)、[pytest.xml](pytest.xml)、[regression.json](regression.json) |
| 正式粒子规模、默认 PPO | **4096 粒子、32 步 rollout、8 步 TBPTT、3 epochs**，真实完成 1 次更新；20.509 秒、32 world steps、131072 agent decisions | [default4096.json](default4096.json) |
| 默认更新 CPU 内存 | 进程峰值 595976 KiB，约 **582 MiB**；不是显存 | [default4096.json](default4096.json) |
| 有界资源探针 | 4096 粒子、4 个测量步平均 124.51 ms；另完成 8 步 rollout/1 epoch 的真实 PPO 更新 | [probe.json](learning-cli/formal_population_bounded_probe/probe.json) |
| 训练与恢复 CLI | contact 训练 2 更新→恢复到累计 3 更新/12 world steps→仅权重 warmstart 到 search→冻结评价，5 个命令全部退出 0 | [learning-cli.json](learning-cli.json)、[逐命令输出](learning-cli/) |
| 短窗评价 | 32 粒子、2 个请求，每个仅 3 步；**2 censored、0 搬运成功**，保留完整分母和逐步轨迹 | [评价摘要](learning-cli/frozen_short_evaluation/summary.json)、[manifest](learning-cli/frozen_short_evaluation/manifest.json) |
| 接触物理诊断 | 固定推力、3 秒模拟：1 个推动者位移 0；8 个推动者物体 x 位移 0.13925。是物理诊断，**不是训练后的合作策略** | [environment.json](environment.json) |
| 真实 Chromium / Canvas | **4096 粒子通过**；播放至 step 15、暂停、重置、缩放、粒子选择、模式切换、中文桌面/移动布局；页面/控制台错误 0 | [canvas-browser.json](viewer-final/canvas-browser.json)、[桌面](viewer-final/canvas-initial.png)、[移动](viewer-final/canvas-mobile.png) |
| 显式软件 WebGPU | **4096 粒子通过**：真实粒子/物体像素、step 0→9→重置 0、选择 2048、缩放、切换 Canvas；页面/控制台/运行时错误 0。adapter 是 **google / SwiftShader 软件设备** | [webgpu-browser.json](webgpu-backend-fixed/webgpu-browser.json)、[截图](webgpu-backend-fixed/webgpu-initial.png)、[保留的早期失败](browser-webgpu-final/webgpu-failure.json) |
| 首轮远端 Linux / Windows CI | 两个系统各 **32 passed、3 CUDA skipped**，各完成4096推理与真实 PPO 探针；Canvas通过；当轮软件WebGPU失败促成后续修复 | [首轮CI](https://github.com/lvdacong/npa-swarm-transport/actions/runs/39943357338)、[结果与工件身份](ci-initial.json) |
| 真实 CUDA、物理浏览器 GPU、正式训练收敛 | **未验证** | [本机接续文档](../../docs/LOCAL_HANDOFF.md) |

不同 CPU 探针的 rollout、执行内容和并行负载不同，不能横向拼成“训练吞吐已最优”。完整默认更新没有完成任何回合；其成功数为 0，不能估计正式成功率。各专项和完整回归重叠，测试数不能相加。

## 关键实现已经检查什么

- 空间邻域与小规模全对枚举一致；密集输入不截邻居，跨并行环境不串边；长链连通判定收敛。
- 全规模 search 初态连通、两端均未知；相对局部观测不含绝对坐标/ID/全局目标向量。网络共享参数，消息每控制步最多传一跳。
- 实际接触力推动物体；没有隔空推动、永久接触奖励、连通保护或自动重连。断连继续回合，评价分别报告搬运与拓扑。
- tanh Gaussian 概率与参考一致；GRU/消息参数实际更新；时间和粒子身份没有被随机打乱；超时从实际终态 bootstrap，并清除下一回合记忆。
- CPU checkpoint 恢复后跨 reset 的下一次完整 PPO 更新逐位一致；CUDA 恢复有 3 个专门用例，留本机实际执行。
- 评价保留超时、删失、异常、未开始请求以及原始轨迹；工程异常令 CLI 非零退出，普通任务超时不伪装工程故障。
- HTTP 实际验证身份、ETag/304、控制与终局停留；CUDA `step_ms` 在同步设备后计时。

## 失败与范围边界

首轮完整测试在容器限制下不能创建 localhost socket，得到 30 passed、3 skipped、2 个 HTTP setup error，保存在 [sandbox-pytest.txt](sandbox-pytest.txt)。获准使用本机测试端口后，同一代码完整回归得到 32 passed、3 skipped；没有把这两个 HTTP 测试删除或跳过。

浏览器初次环境遇到 Playwright driver 权限、字体缺失和软件 WebGPU device loss。前两项已处理，离线中文字体及 OFL 许可在 `web/fonts/`。WebGPU 首轮 CI 也复现同一错误，随后独立三角形在不加载项目 JS 时复现：原参数发生 device loss；指定 Vulkan 但禁用 surface 时提交完成却截图空白；指定 Vulkan 并保留正常 surface 后才有真实像素。原始对照见 [最小探针](webgpu-minimal-probe.json) 与 [呈现探针](webgpu-minimal-presentation-probe.json)。这说明已复现问题不依赖项目 JS 生命周期，不能泛化成排除全部潜在代码问题。

修复只调整浏览器脚本：完整 Chromium/new headless，软件测试增加 `--enable-features=Vulkan --use-vulkan=swiftshader`，保留正常 surface。没有降低像素断言或忽略错误。其后完整4096测试通过，旧失败继续保留。该配置用于软件 CI，不应套到用户日常硬件浏览器或冒充物理GPU。

物理是二维过阻尼软圆盘模型，包含阻力、静摩擦和有限推力；没有惯性、抓取、旋转、复杂障碍或真实机器人标定。所有粒子共享模拟坐标轴，无绝对位置输入不等于旋转等变。

连通仅在初态和每个控制步末采样。软惩罚不保证始终连通；要宣称每个物理子步或连续时间都满足，必须另有测量。正式成功要求整个物体进入目的地区域；contact 预热不能替代未知两端的 search。

原生状态经 CUDA→主机→二进制 HTTP→浏览器 GPU，前端只做必要传输、最新快照与缓冲复用，**不宣称端到端零拷贝**。绘制提交率、唯一模拟状态率和物理显示器呈现率需要区分。

## 身份与复现入口

[system.json](system.json) 和 [regression.json](regression.json) 保存实际 Python/Torch、CPU、源码 SHA-256。系统采集发生在初版发布前，因此 git HEAD 仍是初始 README，`git_dirty=true`；这不是发布后干净提交的证据。最终发布提交、CI 运行和远端读回由主 Issue 绑定。

`source_sha256` 逐文件绑定 `src/npa_swarm/*.py`；前端与文档由 Git 提交树绑定。首轮Windows检出自动转换了文本换行，因此字节摘要与Linux不同，测试实际通过；随后加入 `.gitattributes` 统一文本 LF，避免这类跨平台身份歧义。早期探针保留其执行时身份，不改写成后续提交的结果。大型训练检查点和 NPZ 原件在 `runs/`，其 SHA-256/来源见对应报告；版本化证据含小规模评价的逐步 JSON 原件。

本机按 [LOCAL_HANDOFF.md](../../docs/LOCAL_HANDOFF.md) 执行：真实 CUDA 恢复与有界性能 → 有效接触合力的 contact 短段 → 完整 search → 冻结评价与必要多训练种子/holdout → 物理浏览器与持续运行。允许直接修改、优化和复验，不引入额外冻结或许可门槛；没有新学习证据时不机械补满长训。
