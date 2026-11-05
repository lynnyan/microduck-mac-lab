# MicroDuck 实机契约训练与导出

带训练曲线、失败案例、组合 GIF 和视频的完整记录见 [从零训练图文教程](TUTORIAL.md)；模型、视频、训练输出与离线 HTML 不随 Git 仓库上传；文中产物路径表示历史实验位置或本地训练后生成的位置。

本路线让训练、CPU 回放和导出模型遵循同一份控制契约。训练仍在本机 CPU 上执行，结果可导出为运行时所需的 ONNX。兼容的模型格式不等于已在真实机器人上验证。

## 本次已完成的训练与候选产物

从零训练的站立策略位于 `outputs/runtime_stand_v1`，实际完成 800768 条 transition，使用默认三阶段课程。候选模型为 `outputs/runtime_stand_v1/export/policy.onnx`；它与旧的 `outputs/stand_baseline` 是不同的模型与配方。

| 验证 | 结果 |
| --- | --- |
| 标称物理 | seeds 2027 至 2036，10 次各 20 秒，10 次未跌倒 |
| 传感器误差 | 同样 10 次各 20 秒，10 次未跌倒 |
| 动力学误差 | 同样 10 次各 20 秒，10 次未跌倒 |
| 导出 ONNX 在动力学误差下回放 | 10 次各 20 秒，10 次未跌倒 |
| 导出数值对齐 | 364 组输入，最大动作误差约 1.073e-6，阈值 1e-4 |
| 契约和环境回归 | 10 项通过，含首次续训 rollout 的课程计数 |

结果汇总为 `outputs/runtime_stand_v1/results.json`，完整逐回合数据也在该目录。已有 `preview.mp4` 和 `preview.png` 可检查姿态。这些只覆盖站立和零速度命令；头部命令虽然参与训练，跟踪精度尚未单独验收。行走入口已做短程训练和导出检查，尚未交付收敛的契约版行走策略。

直接回放已生成的候选模型：

```bash
bash scripts/view.sh --onnx outputs/runtime_stand_v1/export/policy.onnx \
  --dr-profile dynamics --seconds 20 --episodes 1
```

## 固定的目标版本

契约为 `contracts/runtime_default_v1.json`，对应官方运行时提交 `d13be2c1a2f4b1e0b0e499b08b92f870518b1384` 的默认参数。机器人模型与 BAM 版本沿用已安装的锁定依赖。

这是对指定源码版本的对齐；尚未读取你的真实机器人的固件版本、运行参数或零位标定。若设备与此版本不同，应先生成与设备对应的契约，再训练；不能只修改部署参数去套用已经训练的模型。

| 项目 | 契约 |
| --- | --- |
| 网络接口 | `obs[1,61]` → `actions[1,14]`，float32 |
| 策略关节 | 左腿 5、颈头 4、右腿 5，排除嘴部 |
| 观测 | 角速度、投影重力、相对 HOME 的位置、速度、上次原始动作、13 维命令 |
| 上次动作 | 网络实际输出，尚未经过外部缩放、低通或目标裁剪 |
| 控制周期 | 50 Hz，每周期 4 个 5 ms 物理子步 |
| 站立槽位 | action scale=1.0，BAM firmware kp=160 |
| 行走槽位 | action scale=0.9，BAM firmware kp=200 |
| 低通 | 头部 alpha=0.5，腿部 alpha=0.7；第一帧不滤波 |
| 目标限幅 | 模拟运行时 Safety 的 ±π 舵机行程裁剪；MJCF 另保留真实关节物理限位 |
| 执行器 | XL330、BAM M6、电压控制和负载压降 |

运行时有 15 个物理舵机，策略仅覆盖 14 个；嘴部位于硬件索引 9，右腿硬件索引因此后移一位。训练模型中的被动关节不被当作可控舵机。

原生策略内部输出限幅为 ±2 rad，并在导出图中保留。其输出就是运行时反馈到下一帧的原始动作；运行时不需要再加这个限幅，也不要把动作额外乘以旧版的 0.35。

低通与动作 scale **位于运行时和训练环境中，不折叠进 ONNX**，避免机器人再次应用而产生双重缩放或滤波。ONNX 只包含观测归一化、Actor 和策略自身的输出限幅。

## 立即训练

```bash
cd microduck-mac-lab
bash scripts/train_runtime.sh --task stand --envs 4 \
  --timesteps 800000 --output outputs/my_runtime_stand
```

不加载旧实验模型，直接从随机初始化训练。默认 curriculum 以所有环境合计的累计 transition 数为单位：

| 累计步数 | 阶段 | 新增误差 |
| --- | --- | --- |
| 0 到 20 万 | nominal | 固定标称物理；reset 关节小扰动 |
| 20 万到 60 万 | sensors | 半强度噪声、编码器偏置、IMU 安装误差、IMU 与速度测量延迟 |
| 60 万之后 | dynamics | 全强度传感误差，加电池电压、压降、BAM 摩擦、质量惯量、质心、足地摩擦和执行延迟 |

阶段在回合 reset 时生效，回合内物理参数保持不变。控制契约始终不变。默认每个训练回合最长 10 秒，失败后正常重置；这不是起身任务。

### 推荐的逐阶段实验

下面的脚本把每个阶段保存成独立目录，便于比较和回退。输出路径必须是新的：

```bash
bash scripts/train_runtime_staged.sh outputs/my_staged_stand
```

它依次训练标称 20 万步、追加传感器阶段 40 万步、追加动力学阶段 40 万步，然后导出最终模型。每次续训保留模型、优化器和归一化，拒绝旧版实验模型或不一致的契约。

将任务改为行走：

```bash
MICRODUCK_TASK=walk bash scripts/train_runtime_staged.sh outputs/my_staged_walk
```

这些预算是起始实验设置，不保证步态收敛。若某阶段的性能下降，先比较标称组和该阶段的评估，再延长训练或减缓随机化；不要通过更改动作含义补偿。

### 手动续训

```bash
bash scripts/train_runtime.sh --task stand \
  --resume outputs/my_staged_stand/nominal --dr-profile sensors \
  --timesteps 400000 --output outputs/my_sensor_extension
```

`--timesteps` 是本次追加预算。task、actuator、n-steps、batch-size 和契约必须兼容。若使用自动 curriculum，进度会按已恢复的累计步数继续，避免每次续训又从无误差阶段开始。

## 误差范围与控制命令

动力学阶段的起始范围为：电压 6.5 到 8.2 V，压降增益 0 到 0.2，BAM 摩擦预算 0.9 到 1.1 倍，躯干质量与惯量一起缩放 0.95 到 1.05 倍，躯干质心 ±3 mm，armature 0.9 到 1.1 倍，足地摩擦 0.7 到 1.3。执行延迟为 **3 到 6 个物理子步，即 15 到 30 ms**。

传感器全强度为：编码器偏置 ±0.015 rad，IMU 安装误差最大 6°，gyro 噪声 ±0.03 rad/s，重力方向噪声 ±0.01，位置噪声 ±0.001 rad，速度噪声 ±0.25 rad/s。IMU 延迟为 0 或 1 个控制周期，关节速度延迟 1 个控制周期。所有随机化从基线重新采样，不跨 reset 累积。

这些是待验证的起始误差范围，不是你的设备的实测标定。齿隙、温度变化、地形课程、外力扰动、状态切换和实机故障处理尚未全部复刻。

站立训练 twist 为零；行走训练采样 `vx∈[-0.2,0.3]`、`vy∈[-0.1,0.1]`、`wz∈[-0.5,0.5]`，保留 25% 零命令回合。头部仅训练小范围 HOME 偏移：`±0.05, ±0.05, ±0.07, ±0.015 rad`。身体姿态命令保持全零；不要把这些模型当成已训练身体姿态控制的策略。

## 评估与回放

冻结模型，分别评估三种条件。以下以一个已训练 run 为例：

```bash
.venv/bin/python play.py --run outputs/my_runtime_stand \
  --dr-profile nominal --seconds 20 --episodes 10 --headless \
  --report outputs/my_runtime_stand/eval_nominal.json

.venv/bin/python play.py --run outputs/my_runtime_stand \
  --dr-profile sensors --seconds 20 --episodes 10 --headless \
  --report outputs/my_runtime_stand/eval_sensors.json

.venv/bin/python play.py --run outputs/my_runtime_stand \
  --dr-profile dynamics --seconds 20 --episodes 10 --headless \
  --report outputs/my_runtime_stand/eval_dynamics.json
```

交互窗口使用同样的契约和指定误差条件：

```bash
bash scripts/view.sh --run outputs/my_runtime_stand --dr-profile dynamics \
  --seconds 20 --episodes 1
```

`--head-command` 可指定四个头部姿态偏移，默认零；`--command` 指定三个速度指令。要检验小范围命令跟踪，评估时需显式覆盖这些命令。

## 导出 ONNX

```bash
bash scripts/export_runtime.sh --run outputs/my_runtime_stand \
  --output outputs/my_runtime_stand/export
```

导出器拒绝旧实验模型。输出包括：

| 文件 | 内容 |
| --- | --- |
| `policy.onnx` | 归一化、Actor 与内部动作限幅，接口为 `obs` 和 `actions` |
| `contract.json` | 此策略训练时使用的运行时契约 |
| `deployment_requirements.json` | 槽位、scale、gain、低通和命令支持范围 |
| `manifest.json` | 官方 schema 2 的本地策略说明，标记硬件验证未完成 |
| `validation.json` | 数值对齐误差、阈值和文件校验值 |
| `replay_inputs.npz` | 实际用于对齐检查的原始 61 维观测 |

导出检查覆盖三类环境采集的 300 帧观测与 64 组额外输入；与 `SB3 predict(VecNormalize(obs))` 比较，最大绝对动作误差要求小于 `1e-4`。检查通过只证明导出算术与接口一致，不证明行为可靠。

直接用导出的 ONNX 回放，**不要再手动归一化输入**：

```bash
bash scripts/view.sh --onnx outputs/my_runtime_stand/export/policy.onnx \
  --dr-profile nominal --seconds 20 --episodes 1
```

运行时通过外部控制参数应用 scale 和低通。`manifest.json` 对步态的 `action_scale` 不会自动替你修改 robotd 的 tuning；部署前必须核对 `deployment_requirements.json`，不能只复制 ONNX。

## 验证和后续实机流程

```bash
.venv/bin/python test_runtime_contract.py
```

检查包括源码契约的数值案例、嘴部排除映射、原始动作反馈、低通首帧处理、动作延迟单位、课程不改变控制语义、随机化不累积、seed 重置、Gym 接口和旧模型导出拒绝。对齐目标是指定运行时源码的常规 stand/walk 控制段；本机尚未运行 Rust daemon 或连接真实舵机。

形成候选模型后，按顺序完成：多 seed 仿真和命令覆盖、导出数值对齐、原生 ONNX 回放、真实运行时的离线重放，再核对实机固件、参数、关节零位、IMU 坐标和时序。首次实机测试使用支撑和急停。当前任务只生成本地训练与导出产物，没有连接硬件或发布到 Hub。

## 对照源码

- [观测与原始动作反馈](https://github.com/pollen-robotics/microduck/blob/d13be2c1a2f4b1e0b0e499b08b92f870518b1384/duck-control/src/obs.rs)
- [HOME 与嘴部索引](https://github.com/pollen-robotics/microduck/blob/d13be2c1a2f4b1e0b0e499b08b92f870518b1384/duck-control/src/model.rs)
- [缩放、低通和增益选择](https://github.com/pollen-robotics/microduck/blob/d13be2c1a2f4b1e0b0e499b08b92f870518b1384/robotd/src/control.rs)
- [舵机行程裁剪](https://github.com/pollen-robotics/microduck/blob/d13be2c1a2f4b1e0b0e499b08b92f870518b1384/duck-control/src/safety.rs)
- [部署 manifest 语义](https://github.com/pollen-robotics/microduck/blob/d13be2c1a2f4b1e0b0e499b08b92f870518b1384/docs/policy-manifest.md)
