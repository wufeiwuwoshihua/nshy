from openai import OpenAI
import json
import re
import numpy as np
import logging
import requests
import argparse
from tqdm import tqdm


def LLM_response(content):
    response = ""
    return response


def extract_answer(answer):
    match = re.search(r'[A-D]', answer)
    if match:
        return ord(match.group()) - 65
    else:
        return 4


def test(path, type_dataset, log):
    if type_dataset == "folio":
        standard_answer = np.load(".\datasets\FOLIO\\npy\\val.npy")

    elif type_dataset == "pw":
        standard_answer = np.load(".\datasets\pw\\npy\\val.npy")
    else:
        standard_answer = np.load(".\datasets\\ruletaker\\npy\\val.npy")
    with open(path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    answer = np.zeros(len(dataset))
    count = 0
    right = 0
    logging.basicConfig(filename=log, level=logging.INFO)
    dataset = dataset[count:]
    for sample in tqdm(dataset):
        context = sample['context']
        question = "Based on the given context, determine if the following conclusion is true: "
        inference = sample['inference']
        content2 = "Let's think step by step: "
        content = f"Context: {context}\nQuestion: {question}\nConclusion: {inference}\n{content2}"
        answer_res = LLM_response(content)
        response = LLM_response(answer_res  + "\n" + "The conclusion: " + inference + ", is <True/False>:")
        if "true" in response.lower():
            answer[count] = 1
        else:
            answer[count] = 0
        
        if answer[count] == standard_answer[count]:
            right += 1
            logging.info(f"right: {right}, count: {count}")
        logging.info(f"count: {count}")
        logging.info(f"content: {content}")
        logging.info(f"answer_res: {answer_res}")
        logging.info(f"response: {response}")
        logging.info(f"answer: {answer}")
        count += 1

    logging.info(f"right: {right}, count: {count}")
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dataset")
    
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--type', type=str, required=True, help="dataset name")
    parser.add_argument('--log', type=str, required=True, help="log")

    args = parser.parse_args()
    test(args.dataset, args.type, args.log)