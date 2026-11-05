# MicroDuck 室内试验场

这是一个可运行的示例场景，尺寸与摩擦为设计假设，没有进行真实房间测量或地面标定。机器人从 `(0, 0)` 出生，面向 `+X`，所有配置长度单位为米。

| 对象 | 设计 |
| --- | --- |
| 活动区 | 4 × 3 米；X 从 -0.6 到 3.4，Y 从 -1.5 到 1.5 |
| 地面 | 木色外观；滑动摩擦参数 0.8，扭转 0.005，滚动 0.0001，均未标定 |
| 门槛 | 中心 `(0.9, 0)`；沿 X 厚 10 cm，沿 Y 长 120 cm，高 1 cm |
| 箱体 A | 中心 `(1.65, 0.50)`；长宽高 35 × 35 × 25 cm |
| 箱体 B | 中心 `(2.20, -0.45)`；长宽高 45 × 35 × 20 cm |
| 箱体 C | 中心 `(2.80, 0.45)`；长宽高 30 × 50 × 30 cm |
| 边界 | 高 30 cm 的固定围墙 |
| 目标标记 | `(3.05, 0)`，只有显示用途，不触发成功或产生奖励 |

门槛两侧可绕行，箱体交错放置，中间留有通路。箱体与围墙均为固定物体，不模拟推动。木色贴图不会自动产生木地板物理特性。场景沿用官方 `robot_walk.xml` 的机器人碰撞简化，不代表完整机身碰撞精度。

## 生成与查看

在项目 `microduck_sim` 目录执行：

```bash
.venv/bin/python scripts/build_indoor_scene.py --preview
bash scripts/view.sh --run outputs/runtime_stand_v1 --scene scenes/indoor_lab.xml --seconds 10 --episodes 1
```

第一条命令生成 `scenes/indoor_lab.xml`、`outputs/indoor_lab/indoor_lab_preview.png` 和 `indoor_lab_plan.png`。预览需要绘图依赖，可通过 `.tools/uv sync --locked --group tutorial` 安装。生成的 XML 引用本机上游网格的绝对路径；移动到其他机器后重新执行生成命令即可。

查看交互窗口时可用鼠标缩放、旋转相机，观察整片场地。默认回放相机跟随机器人，俯视总览见生成的 PNG。

## 修改参数

键盘驾驶可运行 `bash scripts/drive_indoor.sh`，默认加载已下载的官方 `models/alpha_walking.onnx`，不进行训练。独立驾驶窗口标题为 `MicroDuck Drive | WASD controls robot`，不会显示 MuJoCo 原生控制面板，W 不再切换线框。点击驾驶窗口后，W/S 设置前进/后退，A/D 设置左转/右转，Q/E 设置向左/向右侧移，空格将全部速度命令清零。方向以机器人自身为准。按键设置的速度松开后仍保留，各轴可组合；直线行走前先按空格清除转向。左上角显示当前速度命令；切换到其他窗口时自动清零，Esc 或关闭窗口退出，每次跌倒后进入下一回合并清零命令。命令清零不保证机器人立刻停止或保持平衡；策略未在该场地训练，不能保证越槛或避障。

修改 `indoor_lab.json` 后重新生成。门槛的 `length_x` 是跨越方向的厚度，`width_y` 是横向长度，`height` 是高度。箱体的 `size_xyz` 是完整尺寸，生成器会转换成 MuJoCo 所需的半尺寸。设置 `height: 0.02` 表示 2 cm。

地面和场景物体的接触优先级为 1，脚部为 0，确保场景接触参数生效。因此这版自定义场景中 `dynamics` 配置的脚部摩擦随机化不会改变地面接触摩擦；需要另行变化场景参数来做地面摩擦实验。其余传感器、电压等随机化保持原逻辑。

## 训练接口

```bash
.venv/bin/python train.py --recipe runtime --task walk --scene scenes/indoor_lab.xml --dr-profile nominal --envs 1 --timesteps 200000 --output outputs/indoor_walk_experiment
```

这条命令启动已有速度跟踪任务在新场景中的训练，不会自动添加避障、越障或导航奖励。训练配置记录场景路径，`play.py --run ...` 会默认使用记录的场景，也可用 `--scene` 显式覆盖。场景 XML 仍由调用方保存；配置只记录路径，未打包场景快照。

现有 61 维观测不包含障碍物信息。建议先测试原地站立，再验证直线接近门槛，随后单独设计越槛与绕障任务。需要到达目标、碰撞终止或自主规划时，还应增加对应任务逻辑。`terrain_` 开头的几何体参与足部支撑接触统计；箱体与围墙不作为支撑地面统计。
