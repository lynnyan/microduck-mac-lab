# 上游来源与许可

本仓库包含 MicroDuck Mac 实验的配套代码与文档。未在此为本仓库原创内容指定开源许可证；上游许可证不会自动适用于全部配套代码。

## MicroDuck

- 仓库：https://github.com/pollen-robotics/microduck_rl
- 固定提交：`bae39f5900b318f7e87fc485c30a31213cfa14eb`
- 上游代码：Apache-2.0，详见下载后的 `upstream/LICENSE`。
- 上游机器人模型资产：按 `upstream/LICENSE-MODELS.md` 的 CC BY-NC-SA 4.0 条款使用。

本仓库不复制上游 checkout 或机器人网格。`scripts/fetch_upstream.sh` 下载固定版本，运行前应阅读其许可证。示例中生成的场景 XML 引用下载后的网格，不随仓库提交。

## BAM

- 仓库：https://github.com/Rhoban/bam
- 固定提交：`62bd8ce12154340be97e06f7f41a0ca8f116d967`
- 许可证：Apache-2.0，以该固定提交的许可证为准。依赖通过 `uv.lock` 安装。

## 演示策略

`models/provenance.json` 仅记录官方演示策略的来源、版本和校验值。模型权重不在本仓库内；运行 `scripts/fetch_demo.sh` 时会从原始来源下载。策略及其他第三方依赖各自遵循其来源许可证。

## 文档图片

`docs/assets/architecture.png` 是本项目对话中生成的组件关系图。`tutorial/assets/` 中提交的 PNG 为原实验的静态图表和回放截图；对应模型、视频、GIF 和原始训练记录不随本仓库上传。
