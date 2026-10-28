# 服务器部署指南

## 问题说明
- 训练在服务器上：`/lixj/pyx/FY3G_GPM_V2`
- 数据处理代码在本地：`E:/2026Code/fy3g_gpm_process`
- 需要将修改后的统计脚本部署到服务器

---

## 方案1：上传修改后的文件到服务器（推荐）

### 步骤1：准备要上传的文件

本地已修改的文件：
```
E:/2026Code/fy3g_gpm_process/processing/statistics.py
E:/2026Code/FY3G_GPM_V2/losses/gpm_loss.py
E:/2026Code/FY3G_GPM_V2/trainers/gpm_trainer.py
E:/2026Code/FY3G_GPM_V2/trainers/gpm_pipeline.py
E:/2026Code/FY3G_GPM_V2/configs/gpm_surface_v3.yaml
```

### 步骤2：上传到服务器

```bash
# 假设你使用scp上传（根据实际情况调整用户名和服务器地址）

# 上传统计脚本（如果服务器有fy3g_gpm_process）
scp E:/2026Code/fy3g_gpm_process/processing/statistics.py \
    root@your-server:/lixj/pyx/fy3g_gpm_process/processing/

# 上传训练相关文件
scp E:/2026Code/FY3G_GPM_V2/losses/gpm_loss.py \
    root@your-server:/lixj/pyx/FY3G_GPM_V2/losses/

scp E:/2026Code/FY3G_GPM_V2/trainers/gpm_trainer.py \
    root@your-server:/lixj/pyx/FY3G_GPM_V2/trainers/

scp E:/2026Code/FY3G_GPM_V2/trainers/gpm_pipeline.py \
    root@your-server:/lixj/pyx/FY3G_GPM_V2/trainers/

scp E:/2026Code/FY3G_GPM_V2/configs/gpm_surface_v3.yaml \
    root@your-server:/lixj/pyx/FY3G_GPM_V2/configs/
```

### 步骤3：在服务器上重新生成统计

```bash
# SSH登录服务器
ssh root@your-server

# 激活环境
cd /lixj/pyx/MultiCloud-env
source activate  # 或者 conda activate your-env-name

# 重新生成统计
cd /lixj/pyx/fy3g_gpm_process  # 如果有这个目录
python -m processing.statistics /lixj/pyx/dataset/FY3G_after

# 或者直接运行脚本
python processing/statistics.py /lixj/pyx/dataset/FY3G_after
```

### 步骤4：启动训练

```bash
cd /lixj/pyx/FY3G_GPM_V2
python scripts/train.py --config configs/gpm_surface_v3.yaml
```

---

## 方案2：服务器上没有fy3g_gpm_process项目

如果服务器上没有数据处理项目，有两个选择：

### 选择A：上传整个fy3g_gpm_process项目

```bash
# 压缩整个项目
cd E:/2026Code
tar -czf fy3g_gpm_process.tar.gz fy3g_gpm_process/

# 上传到服务器
scp fy3g_gpm_process.tar.gz root@your-server:/lixj/pyx/

# 在服务器上解压
ssh root@your-server
cd /lixj/pyx
tar -xzf fy3g_gpm_process.tar.gz
```

### 选择B：在服务器上直接创建统计脚本

如果只需要统计功能，我可以为你创建一个独立的统计脚本，不依赖整个项目结构。

---

## 方案3：本地生成统计，上传结果（不推荐）

如果你本地有数据副本，可以：

```bash
# 本地生成统计
cd E:/2026Code/fy3g_gpm_process
python -m processing.statistics E:/path/to/local/FY3G_after

# 上传生成的统计文件到服务器
scp /path/to/FY3G_after/target_statistics.json \
    root@your-server:/lixj/pyx/dataset/FY3G_after/
```

⚠️ 但这要求本地有完整的数据副本，通常不现实。

---

## 检查服务器状态

请先在服务器上检查：

```bash
# 1. 检查是否有fy3g_gpm_process项目
ls -la /lixj/pyx/ | grep fy3g_gpm_process

# 2. 检查数据目录结构
ls -la /lixj/pyx/dataset/FY3G_after/

# 3. 检查当前统计文件
ls -lh /lixj/pyx/dataset/FY3G_after/target_statistics.json

# 4. 检查FY3G_GPM_V2项目是否完整
ls -la /lixj/pyx/FY3G_GPM_V2/
```

---

## 我需要你提供的信息

1. **服务器上是否有 `/lixj/pyx/fy3g_gpm_process` 目录？**
2. **你通常如何上传文件到服务器？** (scp, rsync, git, 其他)
3. **数据路径是否是 `/lixj/pyx/dataset/FY3G_after`？**
4. **是否需要我创建一个独立的统计脚本？**（不依赖整个项目结构）

---

## 快速方案：创建独立统计脚本

如果你想要最简单的方式，我可以创建一个自包含的统计脚本，你只需：
1. 上传这一个文件到服务器
2. 直接运行即可

告诉我是否需要这个方案？
