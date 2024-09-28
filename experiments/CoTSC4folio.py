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

def LLM_response_multi(content):
    res_list = []
    return res_list


def extract_answer(answer):
    match = re.search(r'[A-D]', answer)
    if match:
        return ord(match.group()) - 65
    else:
        return 4


def test(path, type_dataset, log):
    if type_dataset == "folio":
        standard_answer = np.load(".\datasets\FOLIO\\npy\\val.npy")
        # standard_answer = np.load(".\datasets2\FOLIO\\npy\\val.npy")
        # # print(type(option))
    elif type_dataset == "pw":
        standard_answer = np.load(".\datasets\pw\\npy\\val.npy")
        # standard_answer = np.load(".\datasets2\pw\\npy\dev.npy")
    else:
        standard_answer = np.load(".\datasets\\ruletaker\\npy\\val.npy")
    with open(path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    answer = np.zeros(len(dataset))
    count = 0
    right = 0
    logging.basicConfig(filename=log, level=logging.INFO)
    dataset = dataset[count:]
    for sample in tqdm(dataset[:50]):
        context = sample['context']
        question = "Based on the given context, determine if the following conclusion is true: "
        inference = sample['inference']
        content2 = "Let's think step by step: "
        content = f"Context: {context}\nQuestion: {question}\nConclusion: {inference}\n{content2}"
        answer_res = LLM_response_multi(content)
        answer_tmp = [0, 0]
        for i in range(5):
            response = LLM_response(answer_res[i]['message']['content']  + "\n" + "The conclusion: " + inference + ", is <True/False>:")
            
            if "true" in response.lower():
                answer_tmp[0] += 1
            else:
                answer_tmp[1] += 1
        
        if answer_tmp[0] > answer_tmp[1]:
            answer[count] = 1
        else:
            answer[count] = 0
        if answer[count] == standard_answer[count]:
            right += 1
            logging.info(f"right: {right}, count: {count}")
        # print(count)
        # print(content)
        # print(answer_res)
        # print(response)
        # print(answer)
        logging.info(f"count: {count}")
        logging.info(f"content: {content}")
        logging.info(f"answer_res: {answer_res}")
        logging.info(f"response: {response}")
        logging.info(f"answer: {answer}")
        count += 1

    logging.info(f"right: {right}, count: {count}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dataset")
    
    parser.add_argument('--dataset', type=str, required=True, help="数据集的路径")
    parser.add_argument('--type', type=str, required=True, help="数据集名称")
    # parser.add_argument('--start', type=int, required=True, help="开始位置")
    # parser.add_argument('--result', type=str, required=True, help="存储位置")
    # parser.add_argument('--result2', type=str, required=True, help="另一个存储位置")
    parser.add_argument('--log', type=str, required=True, help="log")

    args = parser.parse_args()
    test(args.dataset, args.type, args.log)