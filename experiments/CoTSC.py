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

def format_answers(answers):
    formatted_answers = ""
    for i, answer in enumerate(answers):
        formatted_answers += f"{chr(65 + i)}. {answer} \n"
    return formatted_answers.strip()

def extract_answer(answer):
    match = re.search(r'[A-D]', answer)
    if match:
        return ord(match.group()) - 65
    else:
        return 4

def test(path, type_dataset, log):
    if type_dataset == "reclor":
        standard_answer = np.load(".\datasets\\reclor\\npy\\val.npy")
        # # print(type(option))
    else:
        standard_answer = np.load(".\datasets\logicqa\\npy\\text_eng.npy")
    with open(path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    answer = np.zeros(len(dataset))
    count = 0
    right = 0
    logging.basicConfig(filename=log, level=logging.INFO, encoding='utf-8')
    dataset = dataset[count:]
    for sample in tqdm(dataset[:50]):
        if type_dataset == "reclor":
            # # print(sample)
            option = sample['answers']
            # # print(type(option))
        else:
            option = sample['options'].split('\n')
        context = sample['context']
        question = sample['question']
        content = f"Context: {context}\nQuestion: {question}\nOptions:\n{option}\nChoose the most suitable option, let's think step by step: "
        answer_res = LLM_response_multi(content)
        answer_tmp = [0, 0, 0, 0]
        for i in range(5):
            
            response = LLM_response(answer_res[i]['message']['content'] + "\n" + "Please output the symbol of the answer A/B/C/D: ")
            answer_0 = int(extract_answer(response))
            answer_tmp[answer_0] += 1
        max_number = max(answer_tmp)
    
        # 找到该数第一次出现的位置
        answer[count] = answer_tmp.index(max_number)

        if answer[count] == standard_answer[count]:
            right += 1
            logging.info(f"right: {right}, count: {count+1}")
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
    logging.info(f"right: {right}, count: {count+1}")
        
        
# def main():
#     test(dataset_path)

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