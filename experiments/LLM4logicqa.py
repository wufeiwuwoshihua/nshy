
from formal import Logic_qwitho, Logic_deductive, Logic_Lextract, Logic_simple, Logic_m2one, Logic_qwitho_eng

from manual import load_dataset, LLM_response
import numpy as np
import json
import logging
import argparse
from direct import extract_answer
from tqdm import tqdm

def Logic_Reasoner(context, question, option, dataset):
    answer = []
    response_list = []
    option_list = []
    count = 0
    answer_0 = 0

    if dataset == "logicqa":
        eng_con_prompt = f"Please translate the following title into English, ensuring that no information is lost.\n{context}"
        eng_context = LLM_response(eng_con_prompt)
        eng_ques_prompt = f"Please translate the following title into English, ensuring that no information is lost.\n{question}"
        eng_ques = LLM_response(eng_ques_prompt)
        eng_options_prompt = f"Please translate the following title into English, ensuring that no information is lost.\n{option}"
        eng_options = LLM_response(eng_options_prompt)

    else:
        eng_context = context
        eng_ques = question
        eng_options = option
    
    simple_context = Logic_simple(eng_context)

    logging.info(f"simplified context: {simple_context}")
    
    Lextract_context = Logic_Lextract(simple_context)
    
    logging.info(f"Lextracted context: {Lextract_context}")
    
    for i in range(4):
        # history_message = []
        if dataset == "logicqa":
            proposition = Logic_qwitho(question, option[i])
            eng_pro_prompt = f"Please translate the following title into English, ensuring that no information is lost.\n{proposition}"
            eng_proposition = LLM_response(eng_pro_prompt)
        else:
            eng_proposition = Logic_qwitho_eng(question, option[i])          
        
        reasoning_process = Logic_deductive(Lextract_context, eng_proposition)

        logging.info(f"Reasoning process: {reasoning_process}")
        
        content_2 = f"{reasoning_process}Please carefully read the reasoning process above and summarize whether it considers the proposition to be valid or invalid. If invalid, output 'False' directly. If valid, output 'True' directly."
        response = LLM_response(content_2)
        logging.info(f"response to T/F: {response}")
        if "true" in response.lower():
            answer.append(1)
            count += 1
            response_list.append(reasoning_process)
            if dataset == "logicqa":
                eng_option_prompt = f"Please translate the following title into English, ensuring that no information is lost.\n{option[i]}"
                eng_option = LLM_response(eng_option_prompt)
            else:
                eng_option = option[i]
            option_list.append(eng_option)
        else:
            answer.append(0)
    logging.info(f"init answer: {answer}")
    if count == 1:
        for i in range(4):
            if answer[i] == 1:
                answer_0 = i
    elif count > 1:
        answer_0= Logic_m2one(eng_context, eng_ques, option_list, response_list, eng_options)

        logging.info(f"final answer: {answer_0}")
    return answer, answer_0



def run(path, type_dataset, log):
    if type_dataset == "reclor":
        standard_answer = np.load(".\datasets\\reclor\\npy\\val.npy")
    else:
        standard_answer = np.load(".\datasets\logicqa\\npy\\text_eng.npy")
    dataset = load_dataset(path)
    answer = []
    answer_0 = []
    count = 0
    final_count = 0
    logging.basicConfig(filename=log, level=logging.INFO, encoding='utf-8')
    dataset = dataset[count:]
    for sample in tqdm(dataset):
        if type_dataset == "reclor":
            option = sample['answers']
        else:
            option = sample['options'].split('\n')
        context = sample['context']
        question = sample['question']
        

        answer_res, answer_0_res = Logic_Reasoner(context, question, option, type_dataset)

        if answer_0_res == standard_answer[count]:
 
            final_count += 1
            logging.info(f"final right: {final_count}, total: {count+1}")
                        
        answer.append(answer_res)
        answer_0.append(answer_0_res)

        count += 1
        

    logging.info(f"final right: {final_count}, total: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dataset")
    
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--type', type=str, required=True, help="dataset name")
    parser.add_argument('--log', type=str, required=True, help="log")

    args = parser.parse_args()
    
    run(args.dataset, args.type, args.log)