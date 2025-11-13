# https://github.com/modelscope/evalscope/blob/main/evalscope/benchmarks/mmlu_pro/mmlu_pro_adapter.py
from datasets import load_dataset

PREFIX = "Answer the following multiple choice question "
SUFFIX = "\nThink step by step and then show your choice in the answer field of the following format: 'answer: X', where X is the choice letter."

def preprocess(test_df):
    res_df = []
    categories = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        category = each["category"].replace(' ', '_')
        if category not in res:
            res[category] = []
            categories.append(category)
        res[category].append(each)
    return res, categories

def format_mmlupro_question(single_question):
    category = single_question["category"]
    question = single_question["question"]
    options  = single_question["options"]
    prompt = "about {}.\n\n{}\n\n".format(category, question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choice_map[i], opt)
    return PREFIX + prompt + SUFFIX

if __name__ == "__main__":
    test_df, categories = preprocess(load_dataset('TIGER-Lab/MMLU-Pro', split="test"))
    for category in categories:
        line_texts = []
        for single_question in test_df[category]:
            prompt = format_mmlupro_question(single_question)
            line_texts.append(prompt.replace('\n', '\\n').replace('\r', '\\r'))
        with open(f'mmlu-pro-{category}.txt', 'w') as f:
            f.write('\n'.join(line_texts))
