import os
import re
import json
import time
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import codecs
import toml
import argparse
import queue
import numpy as np
import copy

LANG = {
    "ZH_CN": "Chinese",
    "PT_BR": "Portuguese",
    "FR_FR": "French",
    "ES_LA": "Spanish",
    "DE_DE": "German",
    }

parser = argparse.ArgumentParser(
	prog="python3 run_openai.py",
	description="Run MMLU Pro Benchmark for  a local LLM  via  OpenAI Compatible API.",
	epilog="Specify  options above  to override  one or more settings from config.",
)
parser.add_argument(
	"-c",
	"--config",
	help="Configuration file. Default=config.toml",
	default="config.toml",
)
parser.add_argument(
	"-u",
	"--url",
	help="server url",
)
parser.add_argument(
	"-d",
	"--dataset",
	help="benchmark dataset",
)
parser.add_argument("-a", "--api", help="api key")
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument(
	"--timeout",
	type=float,
	help="Request timeout in seconds",
)
parser.add_argument("--category", type=str)
parser.add_argument("-p", "--parallel", type=int, help="Number of parallel requests")
parser.add_argument("-v", "--verbosity", type=int, help="Verbosity level 0-2")
parser.add_argument(
	"--log_prompt",
	help="Writes exact prompt and response into log.txt",
	action="store_true",
)
parser.add_argument(
	"--comment", type=str, help="Comment to be included in the final report."
)
args = parser.parse_args()
config = toml.load(open(args.config))
if args.url:
	config["server"]["url"] = args.url
if args.api:
	config["server"]["api_key"] = args.api
if args.model:
	config["server"]["model"] = args.model
if args.timeout:
	config["server"]["timeout"] = args.timeout
if args.category:
	config["test"]["categories"] = [args.category]
if args.parallel:
	config["test"]["parallel"] = args.parallel
if args.verbosity:
	config["log"]["verbosity"] = args.verbosity
if args.log_prompt:
	config["log"]["log_prompt"] = args.log_prompt
if args.comment:
	config["comment"] = args.comment


client = OpenAI(
	base_url=config["server"]["url"],
	api_key=config["server"]["api_key"],
	timeout=config["server"]["timeout"],
)


def log(message):
	print(message)
	with codecs.open(log_path, "a", "utf-8") as file:
		file.write(message + "\n")


def get_chat_completion(messages):
	try:
		response = client.chat.completions.create(
			model=config["server"]["model"],
			messages=messages,
			temperature=config["inference"]["temperature"],
			max_tokens=config["inference"]["max_tokens"],
			top_p=config["inference"]["top_p"],
			frequency_penalty=0,
			presence_penalty=0,
			stop=["Question:"],
			timeout=config["server"]["timeout"],
		)
		try:
			usage_q.put(
				(response.usage.prompt_tokens, response.usage.completion_tokens)
			)
		except:
			pass
		return response.choices[0].message.content.strip()
	except Exception as e:
		print("Resubmitting, Error: ", e)
		time.sleep(3)
		return get_chat_completion(messages)


def get_completion(prompt):
	try:
		response = client.completions.create(
			model=config["server"]["model"],
			prompt=prompt,
			temperature=config["inference"]["temperature"],
			max_tokens=config["inference"]["max_tokens"],
			top_p=config["inference"]["top_p"],
			frequency_penalty=0,
			presence_penalty=0,
			stop=["Question:"],
			timeout=config["server"]["timeout"],
		)
		try:
			usage_q.put(
				(response.usage.prompt_tokens, response.usage.completion_tokens)
			)
		except:
			pass
		if response.choices:
			return response.choices[0].text.strip()
		elif response.content:
			return response.content.strip()
		print("Can't get response.")
		return None
	except Exception as e:
		print("Resubmitting, Error: ", e)
		time.sleep(3)
		return get_completion(prompt)


def load_mmlu_pro(subject):
    # openai/MMMLU
	dataset = load_dataset(args.dataset, subject)
	test_df = dataset["test"]
	test_df = preprocess(test_df, subject)
	return test_df


def preprocess(test_df, subject):
	res_df = []
	for each in test_df:
		each["options"] = [each["A"], each["B"], each["C"], each["D"]]
		each["category"] = subject
		each["answer_index"] = ['A', 'B', 'C', 'D'].index(each["Answer"])
		res_df.append(each)
	res = {}
	for each in res_df:
		if each["Subject"] not in res:
			res[each["Subject"]] = []
		res[each["Subject"]].append(each)
	return res


def format_example(question, options):
	example = "Question: {}\nOptions: ".format(question)
	choice_map = "ABCD"
	for i, opt in enumerate(options):
		example += "{}. {}\n".format(choice_map[i], opt)
	return example.strip()


def chat_prompt_zeroshot(question, options, subject):
	system_prompt = config["inference"]["system_prompt"]
	messages = [
		{
			"role": "system",
			"content": system_prompt.replace("{subject}", subject),
		},
	]
	example = format_example(question, options)
	messages.append({"role": "user", "content": example})
	return messages

def extract_answer(text):
	pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
	match = re.search(pattern, text)
	if match:
		return match.group(1)
	else:
		return extract_again(text)


def extract_again(text):
	pattern = r".*[aA]nswer:\s*\(?([A-J])\)?"
	match = re.search(pattern, text)
	if match:
		return match.group(1)
	else:
		return extract_final(text)


def extract_final(text):
	pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
	match = re.search(pattern, text, re.DOTALL)
	if match:
		return match[0]
	else:
		if config["log"]["verbosity"] >= 1:
			print("Extraction failed:\n", text)
		return None


def run_single_question(single_question):
	question = single_question["Question"]
	options = single_question["options"]
	subject = single_question["Subject"]
	try:
		prompt = chat_prompt_zeroshot(question, options, subject)
		response = get_chat_completion(prompt)
	except Exception as e:
		print("error", e)
		return None, None, None
	pred = extract_answer(response)
	return prompt, response, pred

def update_result(output_res_path, lock):
	category_record = {}
	res = []
	success = False
	while not success:
		try:
			if os.path.exists(output_res_path):
				with lock:
					with open(output_res_path, "r") as fi:
						res = json.load(fi)
						for each in res:
							category = each["category"]
							if category not in category_record:
								category_record[category] = {"corr": 0.0, "wrong": 0.0}
								category_record["random"] = {"corr": 0.0, "wrong": 0.0}
							if not each["pred"]:
								random.seed(12345)
								x = random.randint(0, len(each["options"]) - 1)
								if x == each["answer_index"]:
									category_record[category]["corr"] += 1
									category_record["random"]["corr"] += 1
								else:
									category_record[category]["wrong"] += 1
									category_record["random"]["wrong"] += 1
							elif each["pred"] == each["Answer"]:
								category_record[category]["corr"] += 1
							else:
								category_record[category]["wrong"] += 1
			success = True
		except Exception as e:
			print("Error", e)
	return res, category_record


def evaluate(subjects):
	print("assigned subjects", subjects)
	lock = threading.Lock()
	for subject in subjects:
		test_df = load_mmlu_pro(subject)
		start = time.time()
		print(f"Testing {subject}...")
		system_prompt = config["inference"]["system_prompt"]
		config["inference"]["system_prompt"] = system_prompt.replace(
			"{subject}", "{subject} in " + LANG[subject]
		)
		test_data = []
		for k in test_df:
			# select the last 10 questions from each Subject
			last_10 = len(test_df[k]) - 10
			test_data += test_df[k][last_10:]
		output_res_path = os.path.join(output_dir, subject + "_result.json")
		output_summary_path = os.path.join(output_dir, subject + "_summary.json")
		category_record = {}
		res = []
		with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
			futures = {
				executor.submit(run_single_question, each): each
				for each in test_data
			}
			for future in tqdm(
				as_completed(futures), total=len(futures), smoothing=0.0, ascii=True
			):
				each = futures[future]
				label = each["Answer"]
				category = subject
				prompt, response, pred = future.result()
				if response is not None:
					res, category_record = update_result(output_res_path, lock)
					if category not in category_record:
						category_record[category] = {"corr": 0.0, "wrong": 0.0}
					if config["log"]["log_prompt"]:
						each["prompt"] = prompt
					each["response"] = response
					each["pred"] = pred
					res.append(each)
					if config["log"]["verbosity"] >= 2:
						log_json = {
							"id": each["Unnamed"],
							"question": each["Question"],
							"response": each["response"],
							"pred": each["pred"],
							"answer": each["Answer"],
						}
						print("\n" + json.dumps(log_json, indent="\t"))
					if pred is not None:
						if pred == label:
							category_record[category]["corr"] += 1
						else:
							category_record[category]["wrong"] += 1
					else:
						category_record[category]["wrong"] += 1
					save_res(res, output_res_path, lock)
					save_summary(category_record, output_summary_path, lock)
					res, category_record = update_result(output_res_path, lock)
		save_res(res, output_res_path, lock)
		log(f"Finished testing {subject} in {elapsed(start)}.")
		save_summary(category_record, output_summary_path, lock, report=True)


def save_res(res, output_res_path, lock):
	with lock:
		with open(output_res_path, "w") as fo:
			fo.write(json.dumps(res, indent="\t"))


def print_score(label, corr, wrong):
	try:
		corr = int(corr)
		wrong = int(wrong)
		total = corr + wrong
		acc = corr / total * 100
		log(f"{label}, {corr}/{total}, {acc:.2f}%")
	except Exception as e:
		log(f"{label}, {e} error")


def save_summary(category_record, output_summary_path, lock, report=False):
	total_corr = 0.0
	total_wrong = 0.0
	for k, v in category_record.items():
		if k == "total" or k == "random":
			continue
		cat_acc = v["corr"] / (v["corr"] + v["wrong"])
		category_record[k]["acc"] = cat_acc
		total_corr += v["corr"]
		total_wrong += v["wrong"]
	acc = total_corr / (total_corr + total_wrong)
	category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
	if report:
		print_score("Total", total_corr, total_wrong)
		if "random" in category_record:
			random_corr = category_record["random"]["corr"]
			random_wrong = category_record["random"]["wrong"]
			print_score(
				"Random Guess Attempts",
				random_corr + random_wrong,
				total_corr + total_wrong - random_corr - random_wrong,
			)
			print_score("Correct Random Guesses", random_corr, random_wrong)
			print_score(
				"Adjusted Score Without Random Guesses",
				total_corr - random_corr,
				total_wrong - random_wrong,
			)
	with lock:
		with open(output_summary_path, "w") as fo:
			fo.write(json.dumps(category_record, indent="\t"))


def final_report(assigned_subjects):
	total_corr = 0.0
	total_wrong = 0.0
	random_corr = 0.0
	random_wrong = 0.0
	names = ["overall"] + assigned_subjects
	table = "| " + " | ".join(names) + " |\n"
	separators = [re.sub(r".", "-", name) for name in names]
	table += "| " + " | ".join(separators) + " |\n"
	scores = []
	for file in assigned_subjects:
		res = json.load(open(os.path.join(output_dir, file + "_summary.json")))
		cat_corr = res["total"]["corr"]
		total_corr += cat_corr
		cat_wrong = res["total"]["wrong"]
		total_wrong += cat_wrong
		scores.append(cat_corr / (cat_corr + cat_wrong))
		if "random" in res:
			random_corr += res["random"]["corr"]
			random_wrong += res["random"]["wrong"]
	print_score("Total", total_corr, total_wrong)
	if random_corr and random_wrong:
		print_score(
			"Random Guess Attempts",
			random_corr + random_wrong,
			total_corr + total_wrong - random_corr - random_wrong,
		)
		print_score("Correct Random Guesses", random_corr, random_wrong)
		print_score(
			"Adjusted Score Without Random Guesses",
			total_corr - random_corr,
			total_wrong - random_wrong,
		)
	scores.insert(0, total_corr / (total_corr + total_wrong))
	scores = [f"{score*100:.2f}" for score in scores]
	table += "| " + " | ".join(scores) + " |"
	token_report()
	log("Markdown Table:")
	log(table)


def elapsed(start):
	duration = time.time() - start
	duration_td = timedelta(seconds=duration)
	days = duration_td.days
	hours, remainder = divmod(duration_td.seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	dur_str = ""
	if days:
		dur_str = f"{days} days "
	if hours:
		dur_str += f"{hours} hours "
	if minutes:
		dur_str += f"{minutes} minutes "
	if seconds:
		dur_str += f"{seconds} seconds"
	return dur_str


def token_report():
	ptoks = []
	ctoks = []
	while not usage_q.empty():
		usage = usage_q.get()
		ptoks.append(usage[0])
		ctoks.append(usage[1])
	if ptoks and ctoks:
		log("Token Usage:")
		duration = end - start
		ptoks = np.array(ptoks)
		ctoks = np.array(ctoks)
		log(
			f"Prompt tokens: min {ptoks.min()}, average {ptoks.mean():.0f}, max {ptoks.max()}, total {ptoks.sum()}, tk/s {ptoks.sum()/duration:.2f}"
		)
		log(
			f"Completion tokens: min {ctoks.min()}, average {ctoks.mean():.0f}, max {ctoks.max()}, total {ctoks.sum()}, tk/s {ctoks.sum()/duration:.2f}"
		)


if __name__ == "__main__":
	usage_q = queue.Queue()
	output_dir = "eval_results/" + re.sub(r"\W", "-", config["server"]["model"])
	os.makedirs(output_dir, exist_ok=True)
	log_path = os.path.join(output_dir, "report.txt")
	try:
		os.remove(log_path)
	except:
		pass
	config_copy = copy.deepcopy(config)
	del config_copy["server"]["api_key"]
	del config_copy["test"]["categories"]
	log(f"{datetime.now()}")
	log(json.dumps(config_copy, indent="\t"))
	assigned_subjects = config["test"]["categories"]
	start = time.time()
	evaluate(assigned_subjects)
	end = time.time()
	log(f"Finished the benchmark in {elapsed(start)}.")
	final_report(assigned_subjects)
	print("Report saved to:", log_path)
