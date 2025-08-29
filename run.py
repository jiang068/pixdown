import argparse
from PIL import Image
import numpy as np
import math

def detect_scale(img: Image.Image, block_size=8, tolerance=0.05) -> int:
    """增强容错能力，允许轻微裁切误差"""
    arr = np.array(img)

    # 图像大小
    height, width, _ = arr.shape
    
    # 尝试不同的放大倍数
    possible_scales = []
    for scale in range(2, 20):  # 最大尝试 20倍
        block_h = height // scale
        block_w = width // scale

        # 如果块太小了，跳过
        if block_h < block_size or block_w < block_size:
            continue

        # 计算当前比例下的每个块
        blocks = [arr[y:y + block_h, x:x + block_w] for y in range(0, height, block_h) for x in range(0, width, block_w)]
        
        # 计算每块的平均颜色差异
        block_diffs = []
        for block in blocks:
            mean_color = block.mean(axis=(0, 1))
            diff = np.linalg.norm(block - mean_color, axis=(2))  # 计算颜色差异
            block_diffs.append(np.mean(diff))  # 均值差异

        avg_diff = np.mean(block_diffs)
        
        # 如果差异在容差范围内，认为是合适的放大倍数
        if avg_diff < tolerance * 255:  # tolerance 设置为图像色差的比例
            possible_scales.append(scale)

    # 如果没有找到合适的倍数，则返回 1（即原图）
    if not possible_scales:
        return 1

    # 返回最常见的放大倍数
    return max(set(possible_scales), key=possible_scales.count)

def shrink_pixel_art(input_path, output_path, manual_scale=None):
    img = Image.open(input_path).convert("RGBA")
    
    # 尝试自动检测倍数
    scale = detect_scale(img)
    print(f"自动检测放大倍数: {scale}x")

    # 如果自动检测不准确，可以手动调整
    if manual_scale is not None:
        scale = manual_scale
        print(f"手动修正倍数: {scale}x")

    # 计算新尺寸，确保不压瘪
    target_width = 62
    target_height = 26
    
    # 计算目标尺寸的比例
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)
    new_height = target_height

    # 修正比例，确保宽度/高度不压瘪
    if new_width > target_width:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # 使用最近邻算法进行缩放
    small = img.resize((new_width, new_height), Image.NEAREST)
    small.save(output_path)
    print(f"已保存: {output_path}")

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="像素画缩放工具")
    parser.add_argument('input', type=str, help="输入图片路径")
    parser.add_argument('output', type=str, help="输出图片路径")
    parser.add_argument('--scale', type=int, help="手动设置缩放倍数", default=None)

    args = parser.parse_args()
    
    # 传入命令行参数进行处理
    shrink_pixel_art(args.input, args.output, manual_scale=args.scale)
