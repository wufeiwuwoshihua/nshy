
from formal import Logic_deductive, Logic_Lextract, Logic_simple

from manual import load_dataset, LLM_response
import numpy as np
import json
import logging
import argparse
from direct import extract_answer
from tqdm import tqdm

def Logic_Reasoner(context, inference):

    response_list = []
    count = 0
    answer_0 = 0

    eng_context = context
    
    simple_context = Logic_simple(eng_context)

    logging.info(f"simplified context: {simple_context}")
    Lextract_context = Logic_Lextract(simple_context)
    
    logging.info(f"Lextracted context: {Lextract_context}")
    
    eng_proposition = inference
    
    reasoning_process = Logic_deductive(Lextract_context, eng_proposition)

    logging.info(f"Reasoning process: {reasoning_process}")
    
    content_2 = f"{reasoning_process}Please carefully read the reasoning process above and summarize whether it considers the proposition to be valid or invalid. If invalid, output 'False' directly. If valid, output 'True' directly."
    response = LLM_response(content_2)
    logging.info(f"response to T/F: {response}")

    if "true" in response.lower():
        answer_0 = 1
        count += 1
        response_list.append(reasoning_process)

    else:
        answer_0 = 0

    logging.info(f"answer: {answer_0}")
    
    return answer_0


def run(path, type_dataset, log):
    if type_dataset == "folio":
        standard_answer = np.load(".\datasets\FOLIO\\npy\\val.npy")
    elif type_dataset == "pw":
        standard_answer = np.load(".\datasets\pw\\npy\\val.npy")
    else:
        standard_answer = np.load(".\datasets\\ruletaker\\npy\\val.npy")
    dataset = load_dataset(path)
    count = 0
    final_count = 0
    right_list = []
    logging.basicConfig(filename=log, level=logging.INFO, encoding='utf-8')
    dataset = dataset[count:]
    for sample in tqdm(dataset):
        question = "Based on the given context, determine if the following conclusion is true: "
        content2 = "Directly output <True/False>: "
        if type_dataset == "folio":
            context = sample['context']
            inference = sample['inference']
        elif type_dataset == "pw":
            context = sample['context']
            inference = sample['inference']
        else:
            context = sample['context']
            inference = sample['inference']
        
        answer_0_res = Logic_Reasoner(context, inference)
        if answer_0_res == standard_answer[count]:
            final_count += 1
            logging.info(f"final right: {final_count}, total: {count+1}")
        count += 1
    logging.info(f"right list: {right_list}")
    logging.info(f"final right: {final_count}, total: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dataset")
    
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--type', type=str, required=True, help="dataset name")
    parser.add_argument('--log', type=str, required=True, help="log")

    args = parser.parse_args()
    
    run(args.dataset, args.type, args.log)