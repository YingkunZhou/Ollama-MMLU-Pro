# https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
import random
import pandas

PREFIX = "Answer the following multiple choice question.\n\n"
SUFFIX = "\n\nThink step by step and then show your choice in the answer field of the following format: 'answer: X', where X is the choice letter."

QUERY_TEMPLATE_MULTICHOICE = """
{question}

A. {a}
B. {b}
C. {c}
D. {d}
""".strip()

def format_gpqa_question(single_question, choices):
    question = single_question["Question"]
    a, b, c, d = choices
    prompt = QUERY_TEMPLATE_MULTICHOICE.format(question=question, a=a, b=b, c=c, d=d)
    return PREFIX + prompt + SUFFIX

if __name__ == "__main__":
    df = pandas.read_csv(
        f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    )

    rng = random.Random(0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]

    line_texts = []
    answers = []
    for single_question in examples:
        choices = [
            single_question["Correct Answer"].strip(),
            single_question["Incorrect Answer 1"].strip(),
            single_question["Incorrect Answer 2"].strip(),
            single_question["Incorrect Answer 3"].strip(),
        ]
        choices = [choices[i] for i in single_question["permutation"]]
        prompt = format_gpqa_question(single_question, choices)
        line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))

        correct_index = single_question["permutation"].index(0)
        answers.append("ABCD"[correct_index])


    with open('gpqa-diamond.txt', 'w') as f:
        f.write('\n'.join(line_texts))
    with open('gpqa-diamond.ans', 'w') as f:
        f.write('\n'.join(answers))