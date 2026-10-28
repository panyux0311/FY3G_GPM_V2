# FY3G_GPM_V2

使用过去三个小时的 FY-3G Ku/Ka 雷达观测，预测未来 1、2、3 小时的 GPM IMERG 地表降水率。

本项目负责模型训练、验证、测试及可视化。原始数据读取、质控、重采样和数据集构建由独立的数据处理项目负责。

## 环境安装

在包含 Python 和适配服务器 CUDA 的 PyTorch 环境中安装项目：

```bash
python -m pip install -e .
```

依赖声明见 [pyproject.toml](pyproject.toml)。项目支持 `simvpv2` 与 `unet_standard` 两种模型。

## 实验配置

配置文件名、`config.name`、`train.experiment_group` 和输出目录统一按版本命名，训练超参数沿用各版本原设置。

| 配置 | 实验内容 | 输出目录 |
| --- | --- | --- |
| `configs/gpm_surface_v1.yaml` | 原默认实验，100 轮、峰值学习率 2e-4 | `outputs/gpm_surface_v1` |
| `configs/gpm_surface_v2.yaml` | 学习率 1e-4、早停与逐轮诊断 | `outputs/gpm_surface_v2` |
| `configs/gpm_surface_v3.yaml` | 固定雨强权重、25 轮调度 | `outputs/gpm_surface_v3` |
| `configs/gpm_surface_v4.yaml` | 区域耦合、多时效解码、Hann 拼接 | `outputs/gpm_surface_v4` |

命令行未指定 `--config` 时使用 v1。已有历史输出目录同步改为对应版本名，checkpoint 内的原始实验配置保留作为历史记录。

## 数据

使用已经处理好的 `fy3ggpm_surface_3h_l1only_0p1_v3` 数据集，包含 `source_manifest.json`、`statistics/` 和 NPY shards。每个样本输入为 `[3,43,64,64]`，预测目标为 `[3,1,64,64]`。

数据集、缓存、模型权重和实验输出不提交到 Git。将 YAML 中的 `data.dataset_root` 改为实际数据目录，或使用命令行的 `--data-root`。

## 训练与评估

在项目根目录运行：

```bash
# 真实数据 CPU 单样本预检
python scripts/preflight.py --config configs/gpm_surface_v4.yaml --data-root /path/to/FY3G_after

# 从头训练新版 SimVPv2
python scripts/train.py --config configs/gpm_surface_v4.yaml --data-root /path/to/FY3G_after

# 恢复同一次实验
python scripts/train.py --config configs/gpm_surface_v4.yaml --data-root /path/to/FY3G_after --resume outputs/gpm_surface_v4/checkpoints/last_resume.pt

# 最终测试与可视化，使用验证阶段冻结的阈值
python scripts/finalize.py --config configs/gpm_surface_v4.yaml --data-root /path/to/FY3G_after --cases 3

# 测试
python run_tests.py
```

新版配置启用区域损失权重、概率与雨强耦合、多时效解码、Hann 拼接和可观测区选模。旧实验继续使用对应的原 YAML 和 checkpoint；新版结构需要从头训练。

## 目录

| 目录 | 职责 |
| --- | --- |
| `configs/` | 数据路径、模型和训练参数 |
| `datasets/` | 成品数据读取、输入拼装和训练采样 |
| `models/` | 模型主干及降水输出头 |
| `losses/` | 联合训练损失 |
| `metrics/` | 评估指标和验证阈值搜索 |
| `trainers/` | 训练、EMA、早停、checkpoint 和网格评估 |
| `visualization/` | 预测图与训练曲线 |
| `scripts/` | 训练、预检、评估和部署入口 |
| `tests/` | 回归测试 |
| `docs/` | 使用说明与设计记录 |

完整训练流程和新实验参数见 [区域感知多时效训练说明](docs/zone_coupled_v2.md)，工程约定见 [开发指南](docs/AGENT.md)。

## Git 跟踪

源码、配置、测试和文档纳入版本管理，`.gitignore` 排除本地数据、实验输出、权重、环境和临时文件。运行新的实验不会把大文件自动加入提交。

```bash
git status
git add <修改的源码或配置>
git commit -m "Describe the change"
git push
```
