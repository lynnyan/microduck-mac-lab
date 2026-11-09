# MicroDuck 在 Apple Silicon 上的仿真与 PPO 训练

完整操作过程、真实结果和组合 GIF 见 [从零训练图文教程](TUTORIAL.md)。本 Git 仓库仅交付代码、文档和静态图，不上传模型、视频、GIF、训练输出或离线 ZIP。文中历史模型路径需要自行训练生成；官方演示模型可单独下载。

本项目在 Apple M5 Pro、48 GB 内存、macOS 26.6.2 上完成原实验。新机器需按下方步骤重建依赖。

**训练默认已升级为实机契约配方 `runtime_contract_v1`。** 新训练使用固定 robotd 版本的缩放、滤波、原始动作反馈和舵机增益，支持误差课程及带归一化的 ONNX 导出。请优先阅读 [实机契约训练与导出手册](RUNTIME_TRAINING.md)。旧实验模型保留，使用 `--recipe legacy` 续训，不能冒充新契约模型导出。

本机可运行原生 MuJoCo、BAM M6 XL330 舵机模型、官方 ONNX 策略回放和 CPU PPO 训练。官方 mjlab 的 MuJoCo Warp 训练需要 NVIDIA CUDA；Apple Metal/MPS 不能替代 CUDA。这里另附 Linux CUDA 主机使用的官方训练脚本，未在本机执行 CUDA 训练，也未提交任何云端任务。

## 从 GitHub 克隆并运行

```bash
git clone https://github.com/lynnyan/microduck-mac-lab.git
cd microduck-mac-lab
bash scripts/setup.sh
bash scripts/fetch_demo.sh
.venv/bin/python scripts/build_indoor_scene.py
bash scripts/drive_indoor.sh
```

这是私有仓库，克隆需要具备仓库访问权限。首次安装会下载 Python 依赖及固定版本的上游源码；`fetch_demo.sh` 另外下载官方行走模型。键盘驾驶默认使用该官方模型，不会训练新策略。

- [组件关系与企业及官方产品对比](docs/COMPONENTS.md)
- [室内场景和键盘控制说明](scenes/README.md)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)

下文的 `outputs/...` 是原实验记录或自行训练后的本地产物，仓库未提供这些模型。

## 立即使用

新增的 4 × 3 米室内试验场包含 1 cm 门槛、三个固定箱体和可配置地面参数。场景生成、查看与训练接入方式见 [室内试验场说明](scenes/README.md)；示例尺寸未实测，尚未训练自主越障或避障。

在终端进入项目：

```bash
cd microduck-mac-lab
```

### 打开官方预训练行走策略

```bash
bash scripts/view.sh
```

它打开 MuJoCo 窗口，使用下载的官方策略，给定 `vx=0.3 m/s`，运行 30 秒后关闭。关闭窗口也可提前退出。这里采用固定命令；更改速度或转向可显式传参：

```bash
bash scripts/view.sh --onnx models/alpha_walking.onnx \
  --command 0.3 0 0.2 --seconds 30 --episodes 1
```

命令依次为前向速度、侧向速度、转向角速度，单位为 m/s、m/s、rad/s。该回放使用官方 61 维输入和 `HOME + action`，内置归一化的 ONNX 接收原始观测。

### 打开本机实际训练的站立策略

```bash
bash scripts/view.sh --run outputs/stand_baseline --seconds 30 --episodes 1
```

### 从头训练站立策略

```bash
bash scripts/train_macos.sh --task stand --envs 4 \
  --timesteps 200000 --output outputs/my_stand
```

输出目录需要是新的，防止覆盖已有模型。这里的 timesteps 是全部环境合计的 transition 数；PPO 按完整 rollout 收集，最终步数可能略超过预算。

### 从头训练行走实验

```bash
bash scripts/train_macos.sh --task walk --envs 4 \
  --timesteps 2000000 --output outputs/my_walk
```

`walk` 是本机实验配方：随机采样低速平面运动命令，并保留 25% 的零命令回合。训练 200 万步不保证形成可靠步态，应查看评估指标和回放再决定是否调整奖励。当前交付的本机训练成果是站立模型，行走示范来自官方预训练模型。

### 保存模型与继续训练

每个训练目录保存以下文件：

| 文件 | 用途 |
| --- | --- |
| `model.zip` | 本机 SB3 PPO 策略及优化器状态 |
| `vecnormalize.pkl` | 观测和奖励归一化统计，回放时必须匹配模型 |
| `checkpoint.zip` | 最近一次周期 checkpoint |
| `checkpoint_vecnormalize.pkl` | 与周期 checkpoint 同时保存的归一化统计 |
| `config.json` | 任务、执行器、seed、设备和上游版本 |
| `training_status.json` | 实际累计训练步数和是否被 Ctrl+C 中断 |
| `progress.csv` | PPO 优化和 rollout 指标 |
| `events.out.tfevents.*` | TensorBoard 日志 |
| `monitor/` | 各环境的回合奖励和长度 |

使用最终模型继续训练，保存到另一个目录：

```bash
bash scripts/train_macos.sh --recipe legacy --task stand \
  --resume outputs/stand_baseline --envs 4 \
  --timesteps 200000 --output outputs/stand_continued
```

续训需要相同 task、actuator、n-steps 和 batch-size，`--timesteps` 表示追加预算。默认 `n-steps=256`、`batch-size=256`。Ctrl+C 会保存当前策略和归一化后退出。

### 查看日志

```bash
bash scripts/tensorboard.sh
```

在浏览器访问 `http://127.0.0.1:6006`。服务仅监听本机回环地址，按 Ctrl+C 停止。

### 不开窗口评估

```bash
.venv/bin/python play.py --run outputs/stand_baseline \
  --seconds 20 --episodes 10 --headless \
  --report outputs/stand_baseline/evaluation_new.json
```

回放会冻结归一化统计并使用确定性动作。评估 JSON 包含每个 seed 的存活、时长、速度和转向 RMSE 及世界 x 位移。`survived` 表示达到给定时长且未触发失败；提前关窗不计作成功。

### 录制视频

```bash
bash scripts/view.sh --run outputs/stand_baseline \
  --seconds 5 --episodes 1 --video outputs/my_stand.mp4
```

视频模式进行离屏渲染，不弹仿真窗口。Mac 的交互窗口需要 `mjpython`，`view.sh` 已处理。无图形的训练和评估直接使用 `.venv/bin/python`。

## 已完成的本机验证

| 检查 | 实际结果 |
| --- | --- |
| MuJoCo 模型及 BAM 初始化 | 通过，14 个执行器 |
| 观测和动作接口 | 61 维 / 14 维 |
| Gymnasium/SB3 环境接口检查 | 通过 |
| 相同 seed 重置一致性 | 通过 |
| 2048 步短训 | 完成，`outputs/smoke` |
| 从头训练站立 | 完成 200704 条 transition，`outputs/stand_baseline` |
| 从最终模型继续训练 | 完成至 201728 步，`outputs/resume_smoke` |
| 站立模型独立评估 | seeds 2027 至 2031，5 次各 20 秒均未跌倒 |
| 官方行走策略 CPU 回放 | seeds 2027 至 2029，3 次各 10 秒均未跌倒 |
| 原生 GUI | 站立策略运行 15 秒、正常退出，报告已保存 |
| 视频编码 | 640×480、50 fps、H.264，5 秒录像已生成并检查画面 |

详细结果在 `outputs/verification.json`、`outputs/stand_baseline/evaluation.json`、`outputs/pretrained_walk_eval.json`、`outputs/stand_viewer.json`。上述为固定标称物理参数下的小样本检查，不是实机或随机化鲁棒性验收。

官方行走策略在 `vx=0.3` 的本地检查中，10 秒世界 x 位移约 0.96 至 0.99 m，平面速度 RMSE 约 0.227 m/s，转向 RMSE 约 0.70 rad/s。它能运行并行走，但跟踪误差明显，不能把“未跌倒”解释为准确跟踪速度。未训练的 HOME 零动作基线约 1.24 秒倒下；站立成果来自闭环策略，而不是把参考角度固定住。

## 旧版本机实验与官方配方的区别

本节描述原有 `native_cpu_experimental_v1` 模型，例如 `outputs/stand_baseline`。新默认配方的动作缩放、滤波、执行器增益和导出流程见实机契约手册，不适用下表的旧动作缩放。

| 项目 | 本机 CPU 实验版 | 官方 CUDA 版 |
| --- | --- | --- |
| 物理 | 原生 MuJoCo 3.10.0 | MuJoCo Warp |
| 任务组织 | 本地 Gymnasium 环境 | 官方 mjlab 任务 |
| PPO | Stable Baselines3 2.9.2 | 官方 RSL-RL |
| 默认执行器 | 官方 CPU BAM M6，固定 7.4 V、压降 0.1 | BAM，含域随机化和延迟等 |
| Actor 输入 | 61 维布局，头部和身体命令为 0 | 61 维，含任务命令、噪声和延迟 |
| 动作 | 有界输出，缩放 0.35，目标限位 | 官方位置偏移 scale=1.0 |
| 网络 | Actor/Critic 各 128×128 | 官方 512×256×128，特权 Critic |
| 奖励 | 本地简化站立/速度跟踪奖励 | 官方完整奖励与课程 |
| 随机化 | reset 关节位置 ±0.01 rad | 物理、传感器、执行器及扰动随机化 |
| 结果格式 | SB3 `model.zip` 和配对归一化文件 | 官方 `.pt` 和部署 ONNX |

旧版本机实验不复刻官方训练结果，模型不能直接传入上游 ONNX 导出器或机器人运行时。新契约配方通过 `export_runtime.py` 导出；导出兼容性与实机可靠性仍需分别验证。训练中没有连接实机、上传 checkpoint 或发布模型。

## 在 Linux CUDA 主机运行官方配方

以下脚本已准备并做 shell 语法检查；本机没有 CUDA，未实跑其 GPU 训练。

复制项目时保留代码、脚本和 lockfile；排除 Mac 的 `.venv`、`.tools`、`outputs`，可以排除 `upstream` 让脚本重新下载。不要将 Mac 的虚拟环境复制到 Linux。

在 Linux NVIDIA 主机的项目根目录执行：

```bash
bash scripts/setup_cuda.sh
MICRODUCK_ENVS=4096 MICRODUCK_ITERS=5000 bash scripts/train_cuda.sh
```

脚本默认训练 `Mjlab-Velocity-Flat-MicroDuck`，首先执行 64 环境、5 次迭代冒烟训练，通过后开始正式训练。通过环境变量更改任务或预算：

```bash
MICRODUCK_TASK=Mjlab-Velocity-Flat-Backlash-MicroDuck \
MICRODUCK_ENVS=1024 MICRODUCK_ITERS=5000 \
bash scripts/train_cuda.sh
```

在 CUDA 主机上从官方 `.pt` 导出，请在 `upstream/` 下使用其独立环境：

```bash
cd upstream
../.tools/uv run --locked python scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /absolute/path/to/model_4999.pt \
  --onnx-file output.onnx --num-envs 1
```

## 安装与重建

依赖安装在项目 `.venv`；uv 在项目 `.tools`，未修改 shell 启动文件或系统 Python 包。uv 管理的 Python 3.12 运行时可存放在用户级缓存中。

```bash
bash scripts/setup.sh
bash scripts/fetch_demo.sh
```

`setup.sh` 根据 `uv.lock` 安装依赖，读取固定上游 Git 提交，并进行本机检查；`fetch_demo.sh` 下载固定 Hugging Face revision 的示范 ONNX，并校验 SHA256 和输入输出形状。初次下载需要访问 GitHub、PyPI 和 Hugging Face。

常见问题：

- `launch_passive` 或窗口在 Mac 上报错：通过 `bash scripts/view.sh ...` 启动，它会使用 `mjpython`。
- `output already contains a run`：换新的 `--output`；继续训练用 `--resume`，保留原实验。
- 归一化文件缺失：模型必须和同次保存的 `vecnormalize.pkl` 配对。
- `Unsupported platform` 或 CUDA 检查失败：Mac 使用本地 CPU 路线；官方脚本只在 Linux CUDA 主机运行。
- 训练输出回合变长但动作不理想：结合视频与任务误差判断，增加训练步数不保证解决奖励设计问题。
- 演示使用官方 ONNX 时拒绝布局：脚本会检查 61/14、关节顺序、HOME 和 action scale，不接受不匹配模型。

## 来源与许可

- [MicroDuck 官方训练仓库](https://github.com/pollen-robotics/microduck_rl/tree/bae39f5900b318f7e87fc485c30a31213cfa14eb)，固定 `bae39f5900b318f7e87fc485c30a31213cfa14eb`，上游工作区未修改。
- [BAM 执行器模型](https://github.com/Rhoban/bam/tree/62bd8ce12154340be97e06f7f41a0ca8f116d967)，固定 `62bd8ce12154340be97e06f7f41a0ca8f116d967`。
- [官方预训练策略](https://huggingface.co/pollen-robotics/microduck-policies/tree/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448)，版本和校验值见 `models/provenance.json`。
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)、[MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html)。

上游代码与模型资产的许可不同；保留 `upstream/LICENSE` 和 `upstream/LICENSE-MODELS.md`。官方模型网格采用 CC BY-NC-SA 4.0，商业使用前需按原许可核对适用范围。

本文档由 superagent 维护。
