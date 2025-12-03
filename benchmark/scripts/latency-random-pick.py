import random
import sys

def select_lines_by_block(filename, block_size=100, select_per_block=5):
    selected_lines = []
    with open(filename, errors='ignore') as file:
        lines = file.readlines()
        block = []
        line_count = 0

        for line in lines:
            block.append(line)
            line_count += 1

            if line_count % block_size == 0:
                selected_lines += random.sample(block, select_per_block)
                block = []

    return selected_lines

def select_random_lines(filename, n=20):
    with open(filename, errors='ignore') as file:
        lines = file.readlines()
    selected_lines = random.sample(lines, n)
    return selected_lines

if __name__ == "__main__":
    logfiles = []
    num_lines = int(sys.argv[1])
    output_name = sys.argv[2]
    for i in range(3, len(sys.argv)):
        logfiles.append(sys.argv[i])
    random_lines = []
    for logfile in logfiles:
        if logfile == 'mmlu-redux.txt':
            random_lines += select_lines_by_block(logfile)
        else:
            random_lines += select_random_lines(logfile, num_lines)

    with open(output_name, 'w') as f:
        f.write(''.join(random_lines).strip())

