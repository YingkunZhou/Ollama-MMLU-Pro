import re
import sys
from datasets import load_dataset

def extract_mcq(text):
    pattern = r"(?i:Answer)\s*:\s*\$*\s*\\?(?:boxed\s*)?\{?([A-J])\}?\s*\$*"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    pattern = r'[A-J]'  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""

import glob

if __name__ == "__main__":
    logfiles = {}
    if len(sys.argv) > 1:
        logfile = sys.argv[1]
        if logfile[-1] == '*':
            loglist = glob.glob(logfile)
        else:
            loglist = [logfile]
        for l in loglist:
            match = re.search(r'mmlu-pro-(.+)\.log', l)
            assert match, "the log file name must be mmlu-pro-*"
            logfiles[match.group(1)] = l
    else:
        logfiles = {"computer_science": "/tmp/tmp.log"}

    df = load_dataset('TIGER-Lab/MMLU-Pro', split="test")
    score = 0.0
    total = 0.0
    for category, logfile in logfiles.items():
        lines = open(logfile).readlines()
        indexresp = [i for i, line in enumerate(lines) if '>>>>>>>>>>>>>>>>>>>>' in line]
        i = 0
        for single_question in df:
            example_categy = single_question["category"].replace(' ', '_')
            correct_answer = single_question["answer"].strip()
            if category == example_categy:
                if i == len(indexresp) - 1:
                    response = ''.join(lines[indexresp[i]+1:])
                else:
                    response = ''.join(lines[indexresp[i]+1:indexresp[i+1]])

                extracted_answer = extract_mcq(response)
                total += 1.0
                score += 1.0 if extracted_answer == correct_answer else 0.0
                # print(i, extracted_answer, correct_answer)
                i += 1

    print(score/total)