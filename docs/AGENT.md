# AGENT 指南：FY3G_GPM_V2 降水预测模型系统

本指南旨在为后续接手维护、迭代或在该代码库上执行任务的 AI Agent 与开发人员提供统一的系统全景、设计决策、工程规范及避坑准则。

最新配置与训练流程见 [zone_coupled_v2.md](zone_coupled_v2.md)。`gpm_surface_v4.yaml` 从头训练多时效版本；旧 YAML 默认保持旧结构及评估口径。当前训练只写精简 `metrics.json`，完整指标和诊断子集清单保存在 checkpoint，分组和子损失写入日志。下文旧版本的 `history.json`、`diagnostic_subset.json` 不再单独生成。新增选模区域、门控和拼接方式必须在训练/评估/finalize 中保持一致；测试集禁止搜索阈值和背景下采样。

---

## 1. 项目定位与职责边界 (Boundary & Scope)

`FY3G_GPM_V2` 是一个**专职于深度学习模型训练、验证、测试、推理评估与可视化归档**的独立工程。

- **职责所在**：
  - 加载已处理完成的 NPY 碎片格式数据与离线统计量。
  - 构建并训练 FY-3G Ku/Ka 三小时雷达反射率至 GPM IMERG 未来 3 小时地表雨强的预测模型（`UNetStandard` 或 `SimVPv2`）。
  - 执行自适应覆盖区阈值搜索，并评测分类 CSI、雨强 CSI、CC、POD、FAR 等指标。
  - 生成预测对比图、空间分布评估图及可交付的归档压缩包。
- **职责之外（严禁越界）**：
  - **严禁**在此项目中编写或恢复 FY-3G L1 原始 HDF 读取、GPM IMERG 重采样、时空切片构建等数据工程代码。
  - 数据生产、质控、统计量计算完全由同级独立项目 [`E:\2026Code\fy3g_gpm_process`](../fy3g_gpm_process) 负责。
  - 本项目**严禁引入 `h5py`** 等原始数据读取依赖。

---

## 2. 架构设计与目录结构 (Architecture & Layout)

本项目全面对齐 MultiCloud 规范，目录与模块职责高度内聚：

```text
FY3G_GPM_V2/
├── configs/               # 实验配置文件
│   ├── gpm_surface_v1.yaml # 原默认实验，100 轮、峰值学习率 2e-4
│   ├── gpm_surface_v2.yaml # 学习率 1e-4、早停与逐轮诊断
│   ├── gpm_surface_v3.yaml # 固定雨强权重、25 轮调度
│   └── gpm_surface_v4.yaml # 区域耦合、多时效解码与 Hann 拼接
├── datasets/              # 数据加载与张量组装
│   ├── gpm_dataset.py     # 支持 mmap 内存映射读取、显式关闭句柄的 PyTorch Dataset
│   ├── gpm_inputs.py      # 43 通道模型输入特征标准化与拼装
│   ├── samplers.py        # 去重网格抽样、雨强分层采样器
│   └── storage.py         # NPY Shard 读取底层辅助
├── models/                # 神经网络结构定义
│   ├── UNetStandard.py    # 标准 U-Net 主干
│   ├── SimVPv2.py        # SimVPv2/gSTA 时空主干
│   ├── precipitation.py  # 两种模型共享的降水输出接口
│   └── factory.py        # 根据配置选择模型
├── losses/                # 复合加权损失函数
│   └── gpm_loss.py        # 包含 Focal Loss, Soft CSI, 雨强回归损失, FSS, 超额雨强损失
├── metrics/               # 评测指标与阈值搜索
│   ├── gpm_metrics.py     # 分类/雨强 CSI、CC、POD、FAR 等评测指标计算
│   └── thresholds.py      # 按覆盖区（Zone）自适应寻找最佳 CSI 决策阈值
├── trainers/              # 训练主循环与生命周期管理
│   ├── checkpoint.py      # Checkpoint 存储、加载与 EMA 权重管理
│   ├── optimization.py    # 优化器配置与学习率 Cosine 调度
│   ├── evaluation.py      # 验证集与测试集评测循环
│   ├── gpm_trainer.py     # 训练主类
│   └── gpm_pipeline.py    # 单次运行/阶段流水线组织
├── visualization/         # 可视化与图表生成
│   ├── gpm.py             # 预测雨强对比图、单轨案例空间分布图
│   └── curves.py          # 训练损失与指标收敛曲线绘制
├── utils/                 # 工具类
│   ├── config.py          # 基于 dataclass 的类型安全配置解析与校验
│   ├── paths.py           # 跨平台路径推断与目录创建
│   ├── logging.py         # 结构化终端与文件日志记录器
│   ├── target_transform.py# 雨强对数变换与归一化
│   └── reporting.py       # Markdown 最终总结报告生成
├── scripts/               # 顶层 CLI 入口与部署脚本
│   ├── train.py           # 模型训练入口
│   ├── preflight.py       # 跨平台 / CPU 快速预检脚本
│   ├── evaluate.py        # 离线指标评测入口
│   ├── finalize.py        # 训练收尾：测试集指标计算、典型个例绘制与结果打包
│   ├── package_gpm.py     # 源码部署打包工具
│   ├── run_gpm_training.sh/start_gpm_tmux.sh     # Linux 训练脚本与 tmux 后台托管
│   └── run_gpm_finalize.sh/start_gpm_finalize_tmux.sh # Linux 收尾评估脚本
├── tests/                 # 跨平台单元测试套件
│   ├── test_model.py
│   ├── test_loss.py
│   ├── test_metrics.py
│   ├── test_dataset.py
│   ├── test_data_contract.py
│   ├── test_finalize.py
│   └── test_training_config.py
├── run_tests.py           # 基于标准库 unittest 的独立测试驱动脚本
├── schema.py              # 数据契约常量与校验函数
├── requirements.txt       # 纯净运行依赖清单
└── pyproject.toml         # 包定义与构建配置
```

---

## 3. 数据契约与张量语义 (Data Contract & Tensor Semantics)

### 3.1 输入数据结构（来自 `fy3g_gpm_process`）
数据集根目录（当前配置为 `/lixj/pyx/dataset/FY3G_after`）必须包含：
- `source_manifest.json`：包含 `dataset_version: "fy3ggpm_surface_3h_l1only_0p1_v3"`、0.1° 网格配置、样本数及哈希。
- `statistics/input_statistics.json`：FY-3G 反射率各通道均值与标准差。
- `statistics/target_statistics.json`：GPM 雨强变换统计量。
- `shards/`：以 `.npy` 存储的样本碎片（`train`, `val`, `test`）。

### 3.2 43 通道输入组装机制 (`datasets/gpm_inputs.py`)
每个样本包含过去 3 个整点时间步（\(t-2h, t-1h, t-0h\)），每个时间步组装为 43 个通道：
- **通道 0～19（20 通道）**：标准化后的雷达反射率（Ku 波段 10 层 + Ka 波段 10 层，高度层严格固定为 `2, 3, 4, 5, 6, 7, 8, 9, 10, 11 km AGL`）。
- **通道 20～39（20 通道）**：雷达数据有效性二值掩码（Valid Mask）。
- **通道 40～41（2 通道）**：像元纬度与经度归一化坐标（Latitude, Longitude）。
- **通道 42（1 通道）**：轨道飞行方向（升轨/降轨标志 Orbit Direction）。

网络输入张量形状为：`[batch, 3, 43, 64, 64]`。

### 3.3 输出目标张量
模型预测未来 3 个小时（\(+1h, +2h, +3h\)）的地表雨强：
- 形状：`[batch, 3, 1, 64, 64]`。
- 双头输出：
  1. `prob`：降水发生概率（Sigmoid 激活，范围 `[0, 1]`）。
  2. `amount`：未门控雨强回归值（ReLU 激活或对数还原雨强）。
- 最终推理雨强：`pred = torch.where(prob >= threshold, amount, 0.0)`。

---

## 4. 关键设计决策与避坑指南 (Key Invariants & Gotchas)

### 4.1 Checkpoint 必须对应当前数据集及模型
- `model.architecture` 可选 `unet_standard` 或 `simvpv2`，两者共用分类、雨强、覆盖区阈值输出接口。
- 当前实验名为 `gpm_surface_v1`，输出目录为 `outputs/gpm_surface_v1`；切换模型时需使用独立的实验名和输出目录。
- 恢复训练及评估仅接受当前 0.1° v3 数据契约、相同 `manifest_sha256` 和相同模型类型的 checkpoint。

### 4.2 Windows 文件句柄与内存映射安全 (`np.load(..., mmap_mode='r')`)
- 在 Windows 环境下，通过 `mmap_mode='r'` 打开的 `.npy` 数组会在操作系统级别占用文件句柄。如果未显式释放，会导致单元测试或临时目录清理时触发 `PermissionError: [WinError 32] 另一个程序正在使用此文件`。
- **规范**：
  - `FY3G3HDataset` 实现了 `close()` 方法、`__enter__`/`__exit__` 上下文管理器及 `__del__`。
  - 在创建局部 Dataset 访问数据时，务必在用完后调用 `.close()` 或使用 `with FY3G3HDataset(...) as ds:`。

### 4.3 独立的测试运行器 (`run_tests.py`)
- 由于外部运行环境（如最小化 Conda 环境）可能缺少第三方测试框架 `pytest`，本项目在根目录提供了 `run_tests.py`。
- `run_tests.py` 基于 Python 标准库 `unittest` 驱动 `tests/` 目录下的所有测试用例，**无任何额外依赖**，可在任何纯 Python 环境下无缝执行。

### 4.4 低学习率对照实验的早停与诊断
- `configs/gpm_surface_v2.yaml` 复制了默认实验配置，峰值学习率改为 `1e-4`，实验名、分组和输出目录独立；其余训练算法超参数保持不变。该实验应从头训练，不要恢复旧实验约 75 轮的 checkpoint。
- 启动时从训练集固定选取 8 个均匀分布的完整预报时刻，保存到输出目录的 `diagnostic_subset.json`；后续诊断轮次复用这 8 个时刻。训练及验证诊断使用 EMA 模型，另计算验证集 raw 模型对照。`diagnostics` 中的 `train_ema`、`val_ema`、`val_raw` 均使用固定降水概率阈值 `0.5`，记录分组 CC、CSI 等指标。
- `history.json` 记录 `train_losses`、`validation_loss`、`validation_losses`、`validation_by_lead_target_zone`、`diagnostics`。`validation_loss` 是验证 patch 上按当前 epoch 相同损失权重计算的损失；早停依据验证 CSI，只决定何时结束训练，不改变 warmup/cosine 学习率调度。早停判断写入日志，状态随 `last_resume.pt` 保存。测试集仅在最终评估阶段使用。
- raw/EMA 对照增加模型前向开销，可关闭 `train.diagnostics_enabled`，或增大 `train.diagnostics_every_epochs` 降低诊断频率。

### 4.5 六个雨强阈值 CSI 与原分类 CSI
- 验证、诊断和测试评估自动计算 `0.1`、`0.5`、`1`、`2`、`5`、`10 mm/h` 六个雨强阈值，无需在 YAML 中配置。先对重叠 patch 去重合并，再比较**分类概率门控后的最终预测雨强**和 GPM 真实雨强是否分别大于或等于同一物理阈值。
- 评估结果顶层的 `rain_thresholds_mm_h` 列出阈值；`summary.csi_by_threshold` 使用字符串键 `"0.1"`、`"0.5"`、`"1"`、`"2"`、`"5"`、`"10"`。每个 `by_lead_target_zone` 分组行也有 `csi_by_threshold`，并用 `counts_by_threshold` 逐阈值记录 `tp`、`fp`、`fn`、`positive`。每个阈值独立对有限分组 CSI 做宏平均；真实和预测均无事件时 CSI 为 `NaN`，不得当作 0 参与平均。
- 原 `summary.csi` 是 `0.1 mm/h` 降水发生的分类 CSI；新增 `csi_by_threshold["0.1"]` 根据最终门控雨强计算，两者可能不同。`balanced_score`、早停和最佳模型选择仍使用原分类 CSI。FSS 保留为训练损失项，不属于本次新增评测指标。
- 实验输出目录下 `history.json` 的 `validation`、`validation_by_lead_target_zone`、`diagnostics.train_ema/val_ema/val_raw` 均记录新指标；`scripts/evaluate.py` 打印六阈值 CSI 的汇总 JSON，`scripts/finalize.py` 将测试指标保存到输出目录的 `test_evaluation.json`。训练终端和输出目录的 `training.log` 每轮输出六个 `val_CSI@t` 与各分组 `CSI@t`；原 `val_CSI` 仍是分类得分。续训不会自动补算旧 epoch 的指标，仅新完成的 epoch 增加六阈值记录。

---

## 5. 常用开发与运维命令 (CLI Workflows)

### 5.1 本地 CPU 快速预检
在开展耗时的长周期训练前，先执行轻量级预检以验证环境、模型前向传播及损失函数计算：
```bash
python scripts/preflight.py --config configs/gpm_surface_v1.yaml
```

### 5.2 启动模型训练 (Linux / Windows)
```bash
python scripts/train.py \
    --config configs/gpm_surface_v1.yaml
```

### 5.3 断点续训 (Resume Training)
```bash
python scripts/train.py \
    --config configs/gpm_surface_v1.yaml \
    --resume "outputs/gpm_surface_v1/checkpoints/last_resume.pt"
```

独立对照实验从头运行：
```bash
python scripts/train.py --config configs/gpm_surface_v2.yaml
```

### 5.4 结果收尾与归档 (Finalization)
训练完成后，运行 `finalize.py`。该脚本加载训练阶段选出的阈值，在独立测试集上评测最终指标并绘制强降水个例图：
```bash
python scripts/finalize.py \
    --config configs/gpm_surface_v1.yaml \
    --cases 5
```

### 5.5 执行全套单元测试
```bash
python run_tests.py
```
