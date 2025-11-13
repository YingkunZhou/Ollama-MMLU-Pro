# https://github.com/modelscope/evalscope/blob/main/evalscope/benchmarks/mmlu_redux/mmlu_redux_adapter.py
from datasets import load_dataset, get_dataset_config_names

PREFIX = "Answer the following multiple choice question "
SUFFIX = "\n\nThink step by step and then show your choice in the answer field of the following format: 'answer: X', where X is the choice letter."

QUERY_TEMPLATE_MULTICHOICE = """
about {category}.

{question}

A. {a}
B. {b}
C. {c}
D. {d}
""".strip()

def format_mmlu_question(single_question):
    question   = single_question["question"]
    error_type = single_question['error_type']
    choices = single_question['choices']
    target_index_list = [int(single_question['answer'])]
    correct_answer = single_question['correct_answer']
    if error_type == 'no_correct_answer' and correct_answer:
        choices[target_index_list[0]] = correct_answer

    a, b, c, d = choices
    category   = single_question["category"].replace('_', ' ')
    prompt = QUERY_TEMPLATE_MULTICHOICE.format(category=category, question=question, a=a, b=b, c=c, d=d)
    return PREFIX + prompt + SUFFIX

if __name__ == "__main__":
    full_questions = []
    dataset_name = "edinburgh-dawg/mmlu-redux-2.0"
    subjects = get_dataset_config_names(dataset_name)
    for subj in subjects:
        df = load_dataset(dataset_name, subj, split="test")
        full_questions += [example | {'category': subj} for example in df]

    line_texts = []
    for single_question in full_questions:
        prompt = format_mmlu_question(single_question)
        line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))

    with open('tmp.txt', 'w') as f:
        f.write('\n'.join(line_texts))
