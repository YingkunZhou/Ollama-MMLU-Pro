# https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py
import re
import sys
import pandas
from statistic import statistic

def normalize_number(s):
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return match.group(0)

def extract_boxed_text(text):
    pattern = r'boxed{(.*?)}|framebox{(.*?)}'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(',')[-1].strip()
    pattern = r'\d+'  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""

if __name__ == "__main__":
    logfile = '/tmp/tmp.log'
    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    path1 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-I.jsonl"
    df1 = pandas.read_json(path1, lines=True)
    path2 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-II.jsonl"
    df2 = pandas.read_json(path2, lines=True)
    questions = [row.to_dict() for _, row in df1.iterrows()] + [row.to_dict() for _, row in df2.iterrows()]

    # FIXME: why ignore?
    lines = open(logfile, errors='ignore').readlines()
    indexresp = [i for i, line in enumerate(lines) if '>>>>>>>>>>>>>>>>>>>>' in line]
    score = 0.0
    assert len(indexresp) == len(questions)
    for i in range(len(indexresp)):
        if i == len(indexresp) - 1:
            response = ''.join(lines[indexresp[i]+1:])
        else:
            response = ''.join(lines[indexresp[i]+1:indexresp[i+1]])

        extracted_answer = extract_boxed_text(response)
        try: # All AIME answers are integers, so we convert the extracted answer to an integer
            extracted_answer = int(extracted_answer)
        except (ValueError, TypeError):
            extracted_answer = None

        question_answer = questions[i]["answer"]
        correct_answer  = int(normalize_number(question_answer)) if isinstance(question_answer, str) else question_answer

        score += 1.0 if extracted_answer == correct_answer else 0.0

    print(f"score: {score/len(questions)*100:.1f}")
    tokens_list, _ = statistic(logfile)
    assert len(tokens_list) == len(questions)
