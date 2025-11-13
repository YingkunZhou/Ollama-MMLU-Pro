# https://github.com/modelscope/evalscope/blob/main/evalscope/benchmarks/mmlu_redux/mmlu_redux_adapter.py
import re
import sys
from datasets import load_dataset, get_dataset_config_names

def extract_mcq(text):
    pattern = r"(?i:Answer)\s*:\s*\$*\s*\\?(?:boxed\s*)?\{?([A-D])\}?\s*\$*"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    pattern = r'[A-D]'  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""

def extract_ans(single_question):
    error_type = single_question['error_type']
    choices = single_question['choices']
    target_index_list = [int(single_question['answer'])]
    correct_answer = single_question['correct_answer']
    if error_type == 'no_correct_answer' and correct_answer:
        choices[target_index_list[0]] = correct_answer
    elif error_type == 'wrong_groundtruth' and correct_answer:
        try:
            target_index_list = [int(correct_answer)]
        except ValueError:
            choice_index = ord(correct_answer) - ord('A')
            target_index_list = [choice_index]
    elif error_type == 'multiple_correct_answers' and correct_answer:
        correct_answer = correct_answer.strip('()')
        try:
            correct_answer = correct_answer.replace(' and ', ',').replace(' or ', ',')
            target_index_list = list(map(int, correct_answer.split(',')))
        except ValueError:
            try:
                target_index_list = [ord(c) - ord('A') for c in correct_answer.split(',')]
            except TypeError:
                # find the index of the correct answer in choices
                target_index_list = [choices.index(c) for c in correct_answer.split(',') if c in choices]
    return ['ABCD'[i] for i in target_index_list] if target_index_list else ['A', 'B', 'C', 'D']

if __name__ == "__main__":
    logfile = '/tmp/tmp.log'
    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    full_questions = []
    dataset_name = "edinburgh-dawg/mmlu-redux-2.0"
    subjects = get_dataset_config_names(dataset_name)
    for subj in subjects:
        df = load_dataset(dataset_name, subj, split="test")
        full_questions += [example for example in df]

    lines = open(logfile).readlines()
    indexresp = [i for i, line in enumerate(lines) if '>>>>>>>>>>>>>>>>>>>>' in line]

    score = 0.0
    assert len(indexresp) == len(full_questions)
    for i in range(len(indexresp)):
        if i == len(indexresp) - 1:
            response = ''.join(lines[indexresp[i]+1:])
        else:
            response = ''.join(lines[indexresp[i]+1:indexresp[i+1]])

        correct_answer   = extract_ans(full_questions[i])
        extracted_answer = extract_mcq(response)
        score += 1.0 if extracted_answer in correct_answer else 0.0

    print(score/len(full_questions))
