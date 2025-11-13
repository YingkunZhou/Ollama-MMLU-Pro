# https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py
import re
import pandas

# https://huggingface.co/Qwen/Qwen3-0.6B
MATH_TEMPLATE = """
Given the following problem:
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
""".strip()

def format_aime_question(single_question):
    question = single_question["question"]
    return MATH_TEMPLATE.format(question=question)

def normalize_number(s):
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return match.group(0)

if __name__ == "__main__":
    path1 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-I.jsonl"
    df1 = pandas.read_json(path1, lines=True)
    path2 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-II.jsonl"
    df2 = pandas.read_json(path2, lines=True)
    questions = [row.to_dict() for _, row in df1.iterrows()] + [row.to_dict() for _, row in df2.iterrows()]

    line_texts = []
    for single_question in questions:
        prompt = format_aime_question(single_question)
        line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))

    with open('aime2025.txt', 'w') as f:
        f.write('\n'.join(line_texts))
