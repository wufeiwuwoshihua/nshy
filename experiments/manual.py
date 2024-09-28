import json
import re

import requests


def LLM_response(content):
    response = ""
    return response

def LLM_response_multi(content):
    res_list = []
    return res_list

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

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
