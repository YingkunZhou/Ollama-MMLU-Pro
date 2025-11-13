# https://github.com/openai/simple-evals/blob/main/humaneval_eval.py
# https://github.com/modelscope/evalscope/blob/main/evalscope/benchmarks/humaneval/humaneval_adapter.py
from datasets import load_dataset

PREFIX = "You are an expert Python programmer. Read the following function signature and docstring, and fully implement the function described.\n\n"
SUFFIX = "\n\nYour response should only contain the code for this function."

HUMANEVAL_TEMPLATE = """
### Function signature:
{func_sig}

### Docstring:
{docstring}
""".strip()

def format_humaneval_question(single_question):
    question = single_question["prompt"]
    question_split = question.split('"""') if '"""' in question else question.split("'''")
    func_sig  = question_split[0].strip()
    docstring = '"""\n' + question_split[1].strip() + '\n"""'
    prompt = HUMANEVAL_TEMPLATE.format(func_sig=func_sig, docstring=docstring)
    return PREFIX + prompt + SUFFIX

if __name__ == "__main__":
    df = load_dataset("openai_humaneval", split="test")
    line_texts = []
    for single_question in df:
        prompt = format_humaneval_question(single_question)
        line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))

    with open('humaneval.txt', 'w') as f:
        f.write('\n'.join(line_texts))
