from openai import OpenAI
import json
import re
import numpy as np
import logging

import os

import argparse


def LLM_response(content):
    response = ""
    return response

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

def test(path, type_dataset):
    with open(path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    answer = np.zeros(len(dataset))
    count = 0
    logging.basicConfig(filename='direct.log', level=logging.INFO, encoding='utf-8')

    dataset = dataset[count:]
    for sample in dataset:
        if type_dataset == "reclor":
            option = sample['choices']
        else:
            option = sample['options']
        context = sample['context']
        question = sample['question']
        content = f"Context: {context}\nQuestion: {question}\nOptions:\n{option}\nDirectly output the symbol of the most suitable option A/B/C/D:"
        answer_res = LLM_response(content)
        answer[count] = int(extract_answer(answer_res[0]))
        print(count)
        print(content)
        print(answer_res)
        print(answer)
        logging.info(f"count: {count}")
        logging.info(f"content: {content}")
        logging.info(f"answer_res: {answer_res}")
        logging.info(f"answer: {answer}")
        count += 1
    np.save("direct.npy", answer)
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="dataset")
    
    parser.add_argument('--dataset', type=str, required=True, help="数据集的路径")
    parser.add_argument('--type', type=str, required=True, help="数据集名称")
    
    args = parser.parse_args()
    
    test(args.dataset, args.type)