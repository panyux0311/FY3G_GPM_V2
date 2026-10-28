#!/usr/bin/env python3
"""
独立的统计脚本 - 修复后的权重计算
可以直接在服务器上运行，不依赖fy3g_gpm_process项目

使用方法:
    python standalone_statistics.py /path/to/FY3G_after

输出:
    在数据目录下生成 target_statistics.json
"""
import json
import sys
from pathlib import Path
import numpy as np


def compute_statistics(dataset_dir: Path):
    """计算训练集统计并生成修复后的权重"""

    print(f"正在读取数据集清单: {dataset_dir}")
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到清单文件: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 筛选训练集
    train_records = [r for r in manifest["records"] if r["split"] == "train"]
    print(f"训练集记录数: {len(train_records)}")

    # 雨强分箱边界 (mm/h)
    rain_rate_bins = [0.1, 0.5, 1, 2, 5, 10, 20, 40, 80]
    n_bins = len(rain_rate_bins) + 1
    targets = 1  # 只有一个目标通道

    # 初始化统计
    total_samples = 0
    rain_count = np.zeros(targets, dtype=np.int64)
    group_count = np.zeros((targets, n_bins), dtype=np.int64)

    print("正在统计训练集...")
    for i, record in enumerate(train_records):
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i + 1}/{len(train_records)}")

        target_path = dataset_dir / record["target_path"]
        if not target_path.exists():
            print(f"  警告: 目标文件不存在，跳过: {target_path}")
            continue

        # 读取目标数据
        target = np.load(target_path)  # shape: (height, width) 或 (1, height, width)
        if target.ndim == 3:
            target = target[0]  # 取第一个通道

        valid_mask = ~np.isnan(target)
        valid_pixels = valid_mask.sum()
        total_samples += valid_pixels

        # 统计有雨像元
        rain_mask = valid_mask & (target >= 0.1)
        rain_pixels = rain_mask.sum()
        rain_count[0] += rain_pixels

        # 分箱统计
        for bin_idx in range(n_bins):
            if bin_idx == 0:
                # 第一个箱: [0.1, bins[0])
                bin_mask = rain_mask & (target < rain_rate_bins[0])
            elif bin_idx == n_bins - 1:
                # 最后一个箱: [bins[-1], +inf)
                bin_mask = rain_mask & (target >= rain_rate_bins[-1])
            else:
                # 中间箱: [bins[i-1], bins[i])
                bin_mask = rain_mask & (target >= rain_rate_bins[bin_idx - 1]) & (target < rain_rate_bins[bin_idx])

            group_count[0, bin_idx] += bin_mask.sum()

    print(f"\n统计完成!")
    print(f"总有效像元: {total_samples:,}")
    print(f"有雨像元: {rain_count[0]:,} ({rain_count[0]/total_samples*100:.2f}%)")
    print(f"\n各雨强分箱计数:")
    for bin_idx in range(n_bins):
        if bin_idx == 0:
            label = f"[0.1, {rain_rate_bins[0]})"
        elif bin_idx == n_bins - 1:
            label = f"[{rain_rate_bins[-1]}, +inf)"
        else:
            label = f"[{rain_rate_bins[bin_idx-1]}, {rain_rate_bins[bin_idx]})"
        count = group_count[0, bin_idx]
        pct = count / rain_count[0] * 100 if rain_count[0] > 0 else 0
        print(f"  {label:20s}: {count:12,} ({pct:5.2f}%)")

    # ============ 修复后的权重计算 ============
    print("\n正在计算修复后的权重...")
    group_weight = np.zeros_like(group_count, dtype=np.float64)

    for channel in range(targets):
        if rain_count[channel] > 0:
            # 只对有样本的分箱计算权重
            mask = group_count[channel] > 0
            if mask.any():
                smooth = 10.0
                # 平滑的逆频率权重
                weights = np.sqrt(
                    rain_count[channel] / (group_count[channel][mask].astype(np.float64) + smooth)
                )
                # 限制最大权重，避免极端值
                weights = np.minimum(weights, 20.0)
                # 归一化到均值为1
                weights /= weights.mean()
                group_weight[channel][mask] = weights

    print("\n各雨强分箱权重:")
    for bin_idx in range(n_bins):
        if bin_idx == 0:
            label = f"[0.1, {rain_rate_bins[0]})"
        elif bin_idx == n_bins - 1:
            label = f"[{rain_rate_bins[-1]}, +inf)"
        else:
            label = f"[{rain_rate_bins[bin_idx-1]}, {rain_rate_bins[bin_idx]})"
        weight = group_weight[0, bin_idx]
        print(f"  {label:20s}: {weight:.6f}")

    # 构建输出
    statistics = {
        "version": "1.0-fixed-weights",
        "dataset_path": str(dataset_dir),
        "train_samples": len(train_records),
        "total_pixels": int(total_samples),
        "rain_rate_bins": rain_rate_bins,
        "rain_count": rain_count.tolist(),
        "group_count": group_count.tolist(),
        "group_weight": group_weight.tolist(),
    }

    # 保存到数据集目录
    output_path = dataset_dir / "target_statistics.json"
    print(f"\n保存统计结果到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    print("\n✓ 统计完成!")
    print(f"✓ 已生成: {output_path}")
    print("\n关键改进验证:")
    print(f"  - 空分箱权重: {group_weight[0, -1]:.6f} (应为 0.00)")
    print(f"  - 强雨分箱权重: {group_weight[0, 0]:.6f} (应 > 1.5)")
    print(f"  - 权重范围: [{group_weight[group_weight>0].min():.2f}, {group_weight.max():.2f}]")

    return statistics


def main():
    if len(sys.argv) != 2:
        print("使用方法: python standalone_statistics.py /path/to/FY3G_after")
        print("\n示例:")
        print("  python standalone_statistics.py /lixj/pyx/dataset/FY3G_after")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    try:
        compute_statistics(dataset_dir)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
