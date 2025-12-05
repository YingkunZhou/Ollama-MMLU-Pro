import os
import sys
import argparse
from statistic import statistic
import numpy as np
import matplotlib.pyplot as plt

def process_points_in_windows(x_coords, y_coords, window_size=100, k=100, save_figure=False):
    """按窗口处理数据"""
    # 1. 对纵坐标归一化
    y_normalized = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))

    # 2. 分别求平均值
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_normalized)

    # 3. 计算每个点到平均点的欧几里得距离
    distances = np.sqrt((x_coords - x_mean)**2 + (y_normalized - y_mean)**2)

    # 4. 按窗口处理
    n_samples = len(distances)
    selected_indices = []

    for start_idx in range(0, n_samples, window_size):
        end_idx = min(start_idx + window_size, n_samples)
        window_distances = distances[start_idx:end_idx]
        window_indices = np.arange(start_idx, end_idx)

        # 获取当前窗口内前k小的索引
        if len(window_distances) > 0:
            # 按距离排序，取前k个
            sorted_window_indices = window_indices[np.argsort(window_distances)]
            k_actual = min(k, len(sorted_window_indices))
            selected_indices.extend(sorted_window_indices[:k_actual])

    # 5. 绘图（可选）
    if save_figure:
        plt.figure(figsize=(7, 5))
        plt.scatter(x_coords, y_normalized, s=30, c='#1f77b4', alpha=0.8, label='Samples')
        plt.scatter(x_mean, y_mean, s=150, c='#ff7f0e', marker='D',
                   edgecolors='black', linewidth=1, label='Mean')

        # 标记选中的点
        if len(selected_indices) > 0:
            selected_x = x_coords[selected_indices]
            selected_y = y_normalized[selected_indices]
            plt.scatter(selected_x, selected_y, s=50, c='red', marker='x',
                       linewidths=1.5, label='Selected points')

        plt.xlabel('Accept Ratio', fontsize=11)
        plt.ylabel('Tokens (normalized)', fontsize=11)
        plt.title(f'Data Distribution - Window size: {window_size}, Select {k} per window', fontsize=13)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.2, linestyle='-')
        plt.tight_layout()
        plt.savefig('output_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

    return np.array(selected_indices)

def extract_lines_from_txt(input_file, line_indices):

    f = open(input_file, errors='ignore')
    lines = f.readlines()

    invalid_indices = [idx for idx in line_indices if idx < 0 or idx >= len(lines)]
    if invalid_indices:
        print(f"Warning: Invalid indices in {input_file}: {invalid_indices[:10]}...")
        indices = [idx for idx in line_indices if 0 <= idx < len(lines)]
        print(f"Valid count: {len(line_indices)}")

    extracted_lines = [lines[idx].strip() for idx in line_indices]
    return extracted_lines

def main():
    parser = argparse.ArgumentParser(description='Question selection tool with window processing')
    parser.add_argument('-l', '--logfile',
                       help='Log file containing tokens and accept ratio data')
    parser.add_argument('-i', '--input',
                       help='Input text file to extract lines from')
    parser.add_argument('-o', '--output', default="/tmp/tmp",
                       help='Output file for extracted lines')
    parser.add_argument('-k', type=int, default=1,
                       help='Number of samples to select per window')
    parser.add_argument('-w', '--window-size', type=int, default=100,
                       help='Window size for processing data in chunks')
    parser.add_argument('-s', '--save-figure', action='store_true',
                       help='Save scatter plot to file')

    args = parser.parse_args()

    print(f"处理文件:")
    print(f"  Log文件: {args.logfile}")
    print(f"  输入文件: {args.input}")

    # 从log文件获取统计信息
    tokens_list, accept_ratio = statistic(args.logfile)

    if not accept_ratio:
        print(f"Error: No accept ratio data found in {args.logfile}")
        sys.exit(1)

    if len(accept_ratio) != len(tokens_list):
        print(f"Error: Data length mismatch - tokens_list({len(tokens_list)}) != accept_ratio({len(accept_ratio)})")
        sys.exit(1)

    # 转换accept ratio为小数
    accept_array = np.array(accept_ratio) / 100

    print(f"  数据总数: {len(accept_array)}")
    print(f"  窗口大小: {args.window_size}")
    print(f"  每窗口选取数量: {args.k}")

    # 使用窗口处理函数
    selected_indices = process_points_in_windows(
        x_coords=accept_array,
        y_coords=tokens_list,
        window_size=args.window_size,
        k=args.k,
        save_figure=args.save_figure
    )

    print(f"  总选取索引数: {len(selected_indices)}")
    print(f"  选取索引: {selected_indices}")

    # 从input文件中提取内容
    extracted_lines = extract_lines_from_txt(
        input_file=args.input,
        line_indices=selected_indices
    )

    # 写入输出文件
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(extracted_lines))

    print(f"\n总共提取了 {len(extracted_lines)} 行到 {args.output}")
    print(f"窗口处理完成，每个{args.window_size}个数据中选取了{args.k}个最接近平均点的样本")

if __name__ == "__main__":
    main()