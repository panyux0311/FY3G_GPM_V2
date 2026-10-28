# 服务器部署简易指南

## 🎯 目标
在服务器上部署修复后的代码并重新训练

---

## 📦 需要上传的文件

### 1. 独立统计脚本（用于重新生成权重）
```
E:/2026Code/FY3G_GPM_V2/standalone_statistics.py
```

### 2. 训练代码修改
```
E:/2026Code/FY3G_GPM_V2/losses/gpm_loss.py
E:/2026Code/FY3G_GPM_V2/trainers/gpm_trainer.py
E:/2026Code/FY3G_GPM_V2/trainers/gpm_pipeline.py
E:/2026Code/FY3G_GPM_V2/configs/simvpv2_fixed_weights_short_schedule.yaml
```

---

## 🚀 快速部署步骤

### 步骤1：上传文件到服务器

假设你使用scp（根据实际情况调整）：

```bash
# 上传独立统计脚本
scp E:/2026Code/FY3G_GPM_V2/standalone_statistics.py \
    root@9dgn1jcvuri37-0:/lixj/pyx/FY3G_GPM_V2/

# 上传训练代码
scp E:/2026Code/FY3G_GPM_V2/losses/gpm_loss.py \
    root@9dgn1jcvuri37-0:/lixj/pyx/FY3G_GPM_V2/losses/

scp E:/2026Code/FY3G_GPM_V2/trainers/gpm_trainer.py \
    root@9dgn1jcvuri37-0:/lixj/pyx/FY3G_GPM_V2/trainers/

scp E:/2026Code/FY3G_GPM_V2/trainers/gpm_pipeline.py \
    root@9dgn1jcvuri37-0:/lixj/pyx/FY3G_GPM_V2/trainers/

scp E:/2026Code/FY3G_GPM_V2/configs/simvpv2_fixed_weights_short_schedule.yaml \
    root@9dgn1jcvuri37-0:/lixj/pyx/FY3G_GPM_V2/configs/
```

或者使用WinSCP、FileZilla等图形工具上传。

---

### 步骤2：SSH登录服务器

```bash
ssh root@9dgn1jcvuri37-0
```

---

### 步骤3：重新生成统计（关键步骤）

```bash
# 激活你的环境
cd /lixj/pyx/MultiCloud-env
source activate  # 或者 conda activate your-env-name

# 运行统计脚本
cd /lixj/pyx/FY3G_GPM_V2
python standalone_statistics.py /lixj/pyx/dataset/FY3G_after
```

**预期输出**：
```
正在读取数据集清单: /lixj/pyx/dataset/FY3G_after
训练集记录数: XXXX
正在统计训练集...
统计完成!
✓ 已生成: /lixj/pyx/dataset/FY3G_after/target_statistics.json

关键改进验证:
  - 空分箱权重: 0.000000 (应为 0.00)
  - 强雨分箱权重: 1.896543 (应 > 1.5)
  - 权重范围: [0.19, 2.04]
```

---

### 步骤4：启动训练

```bash
cd /lixj/pyx/FY3G_GPM_V2
python scripts/train.py --config configs/simvpv2_fixed_weights_short_schedule.yaml
```

---

### 步骤5：监控训练

```bash
# 实时查看日志
tail -f outputs/gpm_surface_fixed_weights_short_schedule/training.log

# 查看训练摘要
cat outputs/gpm_surface_fixed_weights_short_schedule/training_summary.json
```

---

## ⚠️ 常见问题

### Q1: 统计脚本提示"找不到清单文件"
**原因**: 数据路径不对  
**解决**: 检查 `/lixj/pyx/dataset/FY3G_after/manifest.json` 是否存在

### Q2: 训练时提示"找不到target_statistics.json"
**原因**: 步骤3的统计没有成功  
**解决**: 重新运行步骤3，确认统计文件已生成

### Q3: 想要恢复旧的训练
**重要**: 修改后的代码**不能resume旧的checkpoint**  
**原因**: 权重计算逻辑已变化，不兼容

---

## 📊 如何验证改进效果

### 训练中监控
```bash
# 查看每轮的CSI指标
grep "CSI by threshold" outputs/gpm_surface_fixed_weights_short_schedule/training.log
```

### 训练后对比

对比新旧实验的 `training_summary.json`:

| 指标 | 基线 | 预期新值 |
|------|------|----------|
| CSI@2 | 0.069 | >0.10 |
| CSI@5 | 0.003 | >0.01 |
| ≥5mm/h POD | 0.13% | >1% |

---

## 🔄 如果需要回退

保留旧代码的备份：

```bash
# 在服务器上备份当前代码
cd /lixj/pyx
cp -r FY3G_GPM_V2 FY3G_GPM_V2.backup

# 如果需要回退
rm -rf FY3G_GPM_V2
mv FY3G_GPM_V2.backup FY3G_GPM_V2
```

---

## 📝 部署检查清单

- [ ] 上传standalone_statistics.py
- [ ] 上传4个训练代码文件
- [ ] SSH登录服务器
- [ ] 激活Python环境
- [ ] 运行统计脚本（验证权重正常）
- [ ] 启动训练
- [ ] 监控训练日志

---

**准备好了就可以开始部署！有问题随时问我。**
