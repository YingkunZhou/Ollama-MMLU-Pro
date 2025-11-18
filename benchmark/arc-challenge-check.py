import re
import sys
from datasets import load_dataset

def extract_mcq(text):
    pattern = r"(?i:Answer)\s*:\s*\$*\s*\\?(?:boxed\s*)?\{?([A-E])\}?\s*\$*"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    pattern = r'[A-E]'  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""

if __name__ == "__main__":
    logfile = '/tmp/tmp.log'
    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    df = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    lines = open(logfile).readlines()
    indexresp = [i for i, line in enumerate(lines) if '>>>>>>>>>>>>>>>>>>>>' in line]
    score = 0.0
    assert len(indexresp) == len(df)
    for i in range(len(indexresp)):
        if i == len(indexresp) - 1:
            response = ''.join(lines[indexresp[i]+1:])
        else:
            response = ''.join(lines[indexresp[i]+1:indexresp[i+1]])

        extracted_answer = extract_mcq(response)
        correct_answer = df[i]['answerKey'].strip()
        score += 1.0 if extracted_answer == correct_answer else 0.0

    print(score/len(df))
