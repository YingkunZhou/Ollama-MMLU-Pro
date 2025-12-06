import os
import sys
import argparse
from statistic import statistic
import numpy as np
import matplotlib.pyplot as plt

def process_points(x_coords, y_coords, save_figure=False):
    # 1. 对纵坐标归一化
    y_normalized = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))

    # 2. 分别求平均值
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_normalized)

    # 3. 计算每个点到平均点的欧几里得距离
    distances = np.sqrt((x_coords - x_mean)**2 + (y_normalized - y_mean)**2)
    sorted_indices = np.argsort(distances)

    # 4. 绘图
    plt.figure(figsize=(7, 5))
    plt.scatter(x_coords, y_normalized, s=30, c='#1f77b4', alpha=0.8, label='Samples')
    plt.scatter(x_mean, y_mean, s=150, c='#ff7f0e', marker='D',
            edgecolors='black', linewidth=1, label='Mean')
    plt.xlabel('Accept Ratio', fontsize=11)
    plt.ylabel('Tokens', fontsize=11)
    plt.title('Data Distribution: Tokens vs Acceptance', fontsize=13)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.2, linestyle='-')
    plt.tight_layout()

    if save_figure:
        plt.savefig('output_plot.png', dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

    return sorted_indices

def extract_lines_from_txt(input_file, line_indices, k=None):
    if 0 in line_indices:
        indices = line_indices
    else:
        indices = [i-1 for i in line_indices]

    if k is not None and k > 0:
        indices = indices[:k]

    f = open(input_file, errors='ignore')
    lines = f.readlines()

    invalid_indices = [idx for idx in indices if idx < 0 or idx >= len(lines)]
    if invalid_indices:
        print(f"Warning: Invalid indices in {input_file}: {invalid_indices[:10]}...")
        indices = [idx for idx in indices if 0 <= idx < len(lines)]
        print(f"Valid count: {len(indices)}")

    extracted_lines = [lines[idx].strip() for idx in indices]
    return extracted_lines

def main():
    parser = argparse.ArgumentParser(description='Question selection tool')
    parser.add_argument('-l', '--logfile', nargs='+', default=['benchmark/results/Qwen3-32B-exl3-2.0bpw-spec-sparse/livecodebench-lite.log'])
    parser.add_argument('-i', '--input', nargs='+', default=["benchmark/livecodebench-lite.txt"])
    parser.add_argument('-o', '--output', default="/tmp/tmp")
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('-s', '--save-figure', action='store_true')

    args = parser.parse_args()

    # 检查log文件和input文件数量是否一致
    if len(args.logfile) != len(args.input):
        print(f"Error: Number of log files ({len(args.logfile)}) does not match number of input files ({len(args.input)})")
        sys.exit(1)

    all_extracted_lines = []

    # 处理每个log-input文件对
    for idx, (log_file, input_file) in enumerate(zip(args.logfile, args.input)):
        print(f"\n处理文件对 {idx+1}/{len(args.logfile)}:")
        print(f"  Log文件: {log_file}")
        print(f"  输入文件: {input_file}")

        # 从log文件获取统计信息
        tokens_list, accept_ratio = statistic(log_file)

        assert accept_ratio is not None, f"Error: accept_ratio is None for {log_file}"
        assert len(accept_ratio) > 0, f"Error: accept_ratio is empty for {log_file}"
        assert len(accept_ratio) == len(tokens_list), f"Error: Data length mismatch for {log_file} - tokens_list({len(tokens_list)}) != accept_ratio({len(accept_ratio)})"

        accept_array = np.array(accept_ratio) / 100

        # 处理数据并获取排序索引
        ordered_indices = process_points(accept_array, tokens_list, save_figure=(args.save_figure and idx == 0))
        print(f"  ordered_indices = {ordered_indices}")

        # 从对应的input文件中提取内容
        extracted_lines = extract_lines_from_txt(
            input_file=input_file,
            line_indices=ordered_indices,
            k=args.k
        )

        all_extracted_lines.extend(extracted_lines)
        print(f"  从该文件提取了 {len(extracted_lines)} 行")

    # 写入输出文件
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(all_extracted_lines))

    print(f"\n总共提取了 {len(all_extracted_lines)} 行到 {args.output}")

if __name__ == "__main__":
    main()