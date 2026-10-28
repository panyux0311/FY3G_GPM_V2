# FY3G_GPM_V2: FY-3G & GPM 降水预报模型工程

基于风云三号 G 星（FY-3G）双频降水测量雷达（PMR）输入，预测 GPM IMERG 未来 1~3 小时地表逐小时平均降水率的深度学习训练与评估工程。
本项目遵循 MultiCloud 架构规范，专职负责 **模型训练、验证、测试、推理评估与可视化归档**。当前配置读取 `/lixj/pyx/dataset/FY3G_after` 的 0.1° v3 成品数据。
纯数据工程（原始 HDF 读取、质控、GPM 重采样、时空切片构建与离线统计量）由独立项目 [`fy3g_gpm_process`](../fy3g_gpm_process) 负责。

工程规范请参阅 [AGENT.md](AGENT.md)。最新改进方案的实现、完整训练流程及启动命令见 [区域感知多时效训练说明](zone_coupled_v2.md)。

新增 `configs/simvpv2_zone_coupled_v2.yaml`，启用区域损失权重、概率/雨强耦合、独立强雨头、SimVPv2 多时效解码、Hann 拼接、分区域 FAR 上限和可观测区选模。使用独立输出目录 `outputs/gpm_surface_zone_coupled_v2`，从头训练：

```bash
python scripts/train.py --config configs/simvpv2_zone_coupled_v2.yaml
```

训练记录以当前代码的 `metrics.json` 和 checkpoint 为准；下文旧版本提及的独立 `history.json`、`diagnostic_subset.json` 已分别合并到 `metrics.json` 与 checkpoint 内。新实验用 observable CSI 选模，旧配置继续用全区域宏平均。

## 快速使用

`configs/default.yaml` 保留当前 `gpm_surface_v1` 实验及原有训练超参数；模型可在其 `model.architecture` 中选择。`configs/simvpv2_lr1e4_earlystop.yaml` 是独立的 SimVPv2 对照实验配置，仅将峰值学习率降至 `0.0001`，并开启早停与逐轮诊断，结果写入 `outputs/gpm_surface_lr1e4_earlystop`，不会覆盖原实验。

### 1. CPU 快速预检
```bash
python scripts/preflight.py --config configs/default.yaml
```

### 2. 启动训练
```bash
python scripts/train.py --config configs/default.yaml

# 从头开始运行学习率 1e-4、早停和诊断对照实验
python scripts/train.py --config configs/simvpv2_lr1e4_earlystop.yaml
```

训练时显示样本索引、epoch、batch 和验证进度条。每个 epoch 都执行验证，并将训练损失、学习率和验证指标打印到终端及 `training.log`。`history.json` 记录 `train_losses`、`validation_loss`、`validation_losses`、`validation_by_lead_target_zone` 和 `diagnostics`；早停判断写入日志，状态写入 `last_resume.pt`。Shell 启动脚本另将终端输出保存到 `console.log`。如需关闭进度条，可设置环境变量 `TQDM_DISABLE=1`。

对照实验启动时从训练集固定选取 8 个均匀分布的完整预报时刻，将清单写入输出目录的 `diagnostic_subset.json`，后续诊断轮次复用该子集。训练与验证诊断使用 EMA 模型，另计算验证集 raw 模型对照；`diagnostics` 中的 `train_ema`、`val_ema`、`val_raw` 均使用固定降水概率阈值 `0.5`，记录分组 CC/CSI 等指标。`validation_loss` 在验证 patch 上使用该 epoch 相同的损失权重计算；早停依据验证 CSI 判断是否停止，只控制训练结束时机，不改变原定的 warmup 与 cosine 学习率调度。早停状态随 `last_resume.pt` 保存。此实验应从头训练，不要恢复旧实验约 75 轮的 checkpoint；测试集仅用于最终评估。raw/EMA 对照需要额外的模型前向计算；可通过 `train.diagnostics_enabled: false` 关闭诊断，或增大 `train.diagnostics_every_epochs` 降低诊断频率。

诊断阈值 `0.5` 是降水概率阈值，实际有雨门槛仍为 `0.1 mm/h`；用于选择最佳模型的主验证 CSI 继续使用逐轮搜索的分区域概率阈值。CC 在重叠 patch 去重后，计算最终门控雨强与 GPM 真值的 Pearson 相关系数，各时效/区域分别计算，再对有限 CC 取平均；常量预测或样本不足时显示 `NaN`。分组记录还包含真实降雨比例，`diagnostics` 的 `data_profile` 记录有效像元数、降雨比例和各覆盖区占比，便于对照训练子集与验证集的分布。

`train_loss` 来自训练态的原始模型，`validation_loss` 来自评估态的 EMA 模型；诊断中的 `train_ema` 和 `val_ema` 才采用相同模型、固定概率阈值及损失权重，适合比较趋势。训练诊断子集并不代表完整训练集。早停的 `min_delta` 只控制耐心计数，最佳 checkpoint 仍按实际 CSI 新高保存；`run_status.json` 记录实际轮数、计划轮数及 `early_stopped` 状态。续训保留早停计数；旧 checkpoint 缺少状态时从历史回放，因此对已长期退化的旧运行开启早停可能立即结束。

每次验证、诊断和测试还会自动计算雨强阈值 `0.1`、`0.5`、`1`、`2`、`5`、`10 mm/h` 的 CSI，无需修改 YAML。先合并并去重重叠 patch，再将**分类概率门控后的最终预测雨强**与真实雨强分别按同一个物理阈值判断事件。评估结果顶层的 `rain_thresholds_mm_h` 列出阈值，`summary.csi_by_threshold` 以字符串键 `"0.1"`、`"0.5"`、`"1"`、`"2"`、`"5"`、`"10"` 记录结果；每个 `by_lead_target_zone` 分组行也有 `csi_by_threshold`，其 `counts_by_threshold` 逐阈值记录 `tp/fp/fn/positive`。各阈值分别对有限的分组 CSI 做宏平均；无真实且无预测事件的分组记为 `NaN`，不按零分计入平均。

原有的 `summary.csi` 和终端 `val_CSI` 仍是 `0.1 mm/h` 降水发生的**分类 CSI**，继续用于 `balanced_score`、早停和最佳模型选择。新增 `csi_by_threshold["0.1"]` 用最终门控雨强判断事件，因此可能与分类 CSI 不同。新增指标保存在实验输出目录的 `history.json` 中的 `validation`、`validation_by_lead_target_zone` 及 `diagnostics.train_ema/val_ema/val_raw`；`scripts/evaluate.py` 打印含六阈值 CSI 的汇总 JSON，`scripts/finalize.py` 将测试评估写入输出目录的 `test_evaluation.json`。训练终端与输出目录的 `training.log` 每轮打印 `val_CSI@t` 和各分组 `CSI@t`，其中 `t` 为上述六个雨强阈值。续训时旧 epoch 的历史记录不会自动补算，新完成的 epoch 会记录新增指标。

### 3. 收尾评估与归档
```bash
python scripts/finalize.py --config configs/default.yaml

# 对照实验完成后使用其自身配置与 checkpoint
python scripts/finalize.py --config configs/simvpv2_lr1e4_earlystop.yaml
```

评估和收尾时使用与该次训练相同的配置及对应 checkpoint。两个模型共享降水分类、雨强回归和阈值接口；checkpoint 会校验模型类型。

### 4. 运行单元测试
```bash
python run_tests.py
```
