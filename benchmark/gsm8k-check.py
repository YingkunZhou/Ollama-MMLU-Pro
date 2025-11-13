from datasets import load_dataset
import re
import sys

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

def extract_ans(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer

if __name__ == "__main__":
    logfile = '/tmp/tmp.log'
    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    df = load_dataset("gsm8k", 'main', split="test")
    lines = open(logfile).readlines()
    indexresp = [i for i, line in enumerate(lines) if '>>>>>>>>>>>>>>>>>>>>' in line]
    score = 0.0
    assert len(indexresp) == len(df)
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

        correct_answer = extract_ans(df[i]['answer'])
        score += 1.0 if extracted_answer == correct_answer else 0.0

    print(score/len(df))
