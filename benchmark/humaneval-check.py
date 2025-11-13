# https://github.com/openai/simple-evals/blob/main/humaneval_eval.py
import re
import sys
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from human_eval.execution import check_correctness  # , unsafe_execute
from human_eval.evaluation import estimate_pass_at_k
import numpy as np

def find_code(completion):
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else completion
    extracted_answer = extracted_answer[
        extracted_answer.find(":\n    ") + 2 :
    ]  # remove signature
    return extracted_answer

def evaluate_functional_correctness(
    sample: dict[str, str],
    completions: list[str],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, completion in enumerate(completions):
            args = (sample, completion, timeout, i)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    passed = [int(r["passed"]) for r in results]
    return passed

if __name__ == "__main__":
    logfile = '/tmp/tmp.log'
    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    df = load_dataset("openai_humaneval", split="test")
    total = []
    correct = []
    lines = open(logfile).readlines()
    indexresp = [i for i, line in enumerate(lines) if '>>>>>>>>>>>>>>>>>>>>' in line]
    assert len(indexresp) == len(df)
    for i in range(len(indexresp)):
        if i == len(indexresp) - 1:
            response = ''.join(lines[indexresp[i]+1:])
        else:
            response = ''.join(lines[indexresp[i]+1:indexresp[i+1]])
        results = evaluate_functional_correctness(df[i], [find_code(response)])
        total.append(len(results))
        correct.append(sum(results))
    total = np.array(total)
    correct = np.array(correct)
    pass_at_k = {f'pass@1': estimate_pass_at_k(total, correct, 1).mean()}
    pass_rate = pass_at_k['pass@1']
    print(pass_rate)
