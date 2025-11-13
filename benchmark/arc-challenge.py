from datasets import load_dataset

PREFIX = "Answer the following multiple choice question.\n\n"
SUFFIX = "\n\nThink step by step and then show your choice in the answer field of the following format: 'answer: X', where X is the choice letter."

QUERY_TEMPLATE_MULTICHOICE3 = """
{question}

A. {a}
B. {b}
C. {c}
""".strip()

QUERY_TEMPLATE_MULTICHOICE5 = """
{question}

A. {a}
B. {b}
C. {c}
D. {d}
E. {e}
""".strip()

QUERY_TEMPLATE_MULTICHOICE = """
{question}

A. {a}
B. {b}
C. {c}
D. {d}
""".strip()

def format_arcc_question(single_question):
    question = single_question["question"]
    if len(single_question["choices"]['text']) == 3:
        a, b, c = single_question["choices"]['text']
        prompt = QUERY_TEMPLATE_MULTICHOICE3.format(question=question, a=a, b=b, c=c)
    elif len(single_question["choices"]['text']) == 5:
        a, b, c, d, e = single_question["choices"]['text']
        prompt = QUERY_TEMPLATE_MULTICHOICE5.format(question=question, a=a, b=b, c=c, d=d, e=e)
    else:
        assert len(single_question["choices"]['text']) == 4
        a, b, c, d = single_question["choices"]['text']
        prompt = QUERY_TEMPLATE_MULTICHOICE.format(question=question, a=a, b=b, c=c, d=d)

    return PREFIX + prompt + SUFFIX

if __name__ == "__main__":
    df = load_dataset("ai2_arc", "ARC-Challenge", split="test")

    line_texts = []
    for single_question in df:
        prompt = format_arcc_question(single_question)
        line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))

    with open('arc-challenge.txt', 'w') as f:
        f.write('\n'.join(line_texts))
