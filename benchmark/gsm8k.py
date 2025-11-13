# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-llama.yaml
from datasets import load_dataset

# https://huggingface.co/Qwen/Qwen3-0.6B
MATH_TEMPLATE = """
Given the following problem:
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
""".strip()

def format_gsm8k_question(single_question):
    question = single_question["question"]
    return MATH_TEMPLATE.format(question=question)

if __name__ == "__main__":
    df = load_dataset("gsm8k", 'main', split="test")
    line_texts = []
    for single_question in df:
        prompt = format_gsm8k_question(single_question)
        line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))

    with open('gsm8k.txt', 'w') as f:
        f.write('\n'.join(line_texts))
