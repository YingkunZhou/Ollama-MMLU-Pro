import re

def statistic(logfile):
    decoding_pattern = r"decoding time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second"
    log_content = open(logfile, errors='ignore').read()
    matches = re.findall(decoding_pattern, log_content, re.IGNORECASE)
    tokens_list = []
    for time_ms, tokens, speed in matches:
        tokens_list.append(int(tokens))

    print(f"average tokens/question: {sum(tokens_list) / len(tokens_list):.1f}")

    pattern = r"accept\s*=\s*([\d.]+)%"
    matches = re.findall(pattern, log_content)
    accpet_ratio = [float(match) for match in matches]
    if len(accpet_ratio):
        print(f"average accept ratio: {sum(accpet_ratio) / len(accpet_ratio):.1f}%")
        return tokens_list, accpet_ratio
    else:
        return tokens_list, None