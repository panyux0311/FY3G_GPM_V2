# 快速启动指南

## 改进方案1、3、7已实施完成

### 执行顺序

#### Step 1: 重新生成训练集统计（必需）
```bash
cd E:/2026Code/fy3g_gpm_process
python -m processing.statistics /lixj/pyx/dataset/FY3G_after
```

这会生成新的 `target_statistics.json`，包含修复后的雨强权重。

**验证**: 检查生成的 `target_statistics.json` 中的 `group_weight`，确保不包含异常大值（旧版本有10.8的异常值）。

---

#### Step 2: 运行新实验
```bash
cd E:/2026Code/FY3G_GPM_V2
python scripts/train.py --config configs/gpm_surface_v3.yaml
```

**预期输出文件**:
- `outputs/gpm_surface_v3/training_summary.json` - 训练摘要
- `outputs/gpm_surface_v3/history.json` - 精简的训练历史
- `outputs/gpm_surface_v3/training.log` - 完整日志
- `outputs/gpm_surface_v3/checkpoints/`:
  - `best_csi_0.1_inference.pt` - 有雨分类最佳
  - `best_csi_2_inference.pt` - ≥2 mm/h最佳
  - `best_csi_5_inference.pt` - ≥5 mm/h最佳
  - `best_balanced_inference.pt` - 综合最佳
  - `last_resume.pt` - 用于恢复训练

---

#### Step 3: 对比分析

对比基线实验和新实验的关键指标：

```bash
cd E:/2026Code/FY3G_GPM_V2
python -c "
import json
baseline = json.load(open('outputs/gpm_surface_v2/training_summary.json'))
new = json.load(open('outputs/gpm_surface_v3/training_summary.json'))

print('=== CSI@0.1 (分类) ===')
print(f'基线: {baseline[\"best_csi_0.1\"]:.4f}')
print(f'新实验: {new[\"best_csi_0.1\"]:.4f}')

print('\n=== CSI@2 (强雨) ===')
print(f'基线: {baseline[\"best_by_objective\"].get(\"csi_2\", {}).get(\"score\", 0):.4f}')
print(f'新实验: {new[\"best_by_objective\"][\"csi_2\"][\"score\"]:.4f}')

print('\n=== CSI@5 (强雨) ===')
print(f'基线: {baseline[\"best_by_objective\"].get(\"csi_5\", {}).get(\"score\", 0):.4f}')
print(f'新实验: {new[\"best_by_objective\"][\"csi_5\"][\"score\"]:.4f}')
"
```

---

### 预期改进

如果改进有效，应该看到：

1. **强降水能力提升**
   - CSI@2: 0.069 → **>0.10**
   - CSI@5: 0.003 → **>0.01**
   - ≥5 mm/h POD: 0.13% → **>1%**

2. **泛化能力改善**
   - 验证损失不会在最佳轮次后持续上升
   - 训练-验证CSI差距缩小

3. **学习率充分衰减**
   - 最佳轮次（假设~15轮）学习率应降至峰值的50%左右
   - 基线第16轮仍在97%

---

### 如何选择checkpoint

根据应用场景选择：

| 场景 | 推荐checkpoint | 原因 |
|------|---------------|------|
| 有雨/无雨分类 | `best_csi_0.1_inference.pt` | 优化分类CSI |
| 中雨预报（≥2 mm/h） | `best_csi_2_inference.pt` | 平衡强雨和误报 |
| 强雨预报（≥5 mm/h） | `best_csi_5_inference.pt` | 最大化强雨捕获 |
| 综合应用 | `best_balanced_inference.pt` | 平衡各项指标 |

---

### 下一步实验（如果基线改进有效）

按优先级继续测试：

#### 实验B: 超阈值正负样本平衡
在新配置基础上添加：
```yaml
train:
  exceedance_positive_weight: [10.0, 20.0, 40.0]  # 对应≥5/10/20 mm/h
```

#### 实验C: DropPath正则化
```yaml
model:
  simvpv2_drop_path: 0.05
```

#### 实验D: 分层组批
修改 `datasets/samplers.py` 实现分层采样（需要代码开发）

---

### 故障排除

#### 问题1: 训练时提示"Checkpoint manifest_sha256 mismatch"
**原因**: 使用了旧的统计文件或checkpoint  
**解决**: 确保重新生成统计，不要resume旧实验

#### 问题2: 权重仍然全是0.5
**原因**: 统计文件未更新或路径错误  
**解决**: 
```bash
# 检查统计文件时间戳
ls -lh /lixj/pyx/dataset/FY3G_after/target_statistics.json
# 应该是今天的日期
```

#### 问题3: 训练过程中loss突然变nan
**原因**: 权重过大导致梯度爆炸（罕见）  
**解决**: 如果出现，降低统计计算中的最大权重上限（从20.0降到15.0）

---

### 监控关键指标

训练过程中重点关注：

```bash
# 实时监控验证CSI
tail -f outputs/gpm_surface_v3/training.log | grep "val_CSI"

# 查看学习率变化
grep "learning_rate" outputs/gpm_surface_v3/training.log

# 检查强雨指标
grep "CSI@2\|CSI@5" outputs/gpm_surface_v3/training.log
```

---

### 联系和反馈

如果遇到问题或有改进建议，请在项目仓库提issue，并附上：
- 完整的training.log
- training_summary.json
- 错误堆栈信息（如有）
