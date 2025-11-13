# pip install "datasets==2.19.1"
# credit to https://github.com/modelscope/evalscope/tree/main/evalscope/benchmarks/live_code_bench
from datasets import load_dataset
import base64
import json
import pickle
import zlib

prompt_template='You are an expert Python programmer. Please solve the following programming question.\n### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)'

class CodeGenerationPromptConstants:
    SYSTEM_MESSAGE_GENERIC = 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.'  # noqa: E501

    SYSTEM_MESSAGE_GEMINI = 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Do NOT use system calls like `exit` in the generated program.'  # noqa: E501

    SYSTEM_MESSAGE_DEEPSEEK = 'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science.'  # noqa: E501

    SYSTEM_MESSAGE_MAGIC = 'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n'  # noqa: E501

    SYSTEM_MESSAGE_WIZARD = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'  # noqa: E501

    SYSTEM_MESSAGE_PHIND = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your fixed program within code delimiters, for example:
```python
# YOUR CODE HERE
```"""  # noqa: E501

    SYSTEM_MESSAGE_CODEQWEN = (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user'  # noqa: E501
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = 'You will use the following starter code to write the solution to the problem and enclose your code within delimiters.'  # noqa: E501

    FORMATTING_WITHOUT_STARTER_CODE = 'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.'  # noqa: E501

    PYTHON_FORMAT = '```python\n# YOUR CODE HERE\n```\n\n'

def transform(item):
    # starter_code
    if item['starter_code']:
        format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'  # noqa: E501
        format_prompt += f"```python\n{item['starter_code']}\n```\n\n"
    else:
        format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n'  # noqa: E501
        format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'

    item['format_prompt'] = format_prompt

    # load test cases
    public_test_cases = item['public_test_cases']
    public_test_cases = json.loads(item['public_test_cases'])

    private_test_cases = item['private_test_cases']
    try:
        private_test_cases = json.loads(private_test_cases)
    except Exception as e:  # noqa: F841
        private_test_cases = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(private_test_cases.encode('utf-8'))))
        )

    # load metadata
    metadata = json.loads(item['metadata'])
    evaluation_sample = json.dumps({
        'inputs': [t['input'] for t in public_test_cases + private_test_cases],
        'outputs': [t['output'] for t in public_test_cases + private_test_cases],
        'fn_name': metadata.get('func_name', None),
    })
    item['evaluation_sample'] = evaluation_sample

    return item

if __name__ == "__main__":
    df = load_dataset("livecodebench/code_generation_lite", version_tag="v6", split="test")

    line_texts = []
    for single_question in df:
        record = transform(single_question)
        question_content = record['question_content']
        format_prompt = record['format_prompt']
        prompt = prompt_template.format(question_content=question_content, format_prompt=format_prompt)
        line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))

    with open('livecodebench-lite.txt', 'w') as f:
        f.write('\n'.join(line_texts))
