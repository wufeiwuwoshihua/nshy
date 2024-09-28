
from formal import Logic_qwitho, Logic_deductive, Logic_Lextract, Logic_simple, Logic_m2one, Logic_qwitho_eng, Logic_deductive_noFOL, Logic_nodeductive

from manual import load_dataset, format_answers, LLM_response, extract_answer
import numpy as np
import json
import logging
import argparse
from direct import extract_answer
from tqdm import tqdm

def Logic_Reasoner(context, question, option, dataset):
    # context_extended = context_extend(context, model)
    answer = []
    response_list = []
    option_list = []
    count = 0
    answer_0 = 0
    # response
    if dataset == "logicqa":
        eng_con_prompt = f"请你将下面的题目翻译成英文，注意不要损失任何信息\n{context}"
        eng_context = LLM_response(eng_con_prompt)
        eng_ques_prompt = f"请你将下面的问题翻译成英文，注意不要损失任何信息\n{question}"
        eng_ques = LLM_response(eng_ques_prompt)
        eng_options_prompt = f"请你将下面的句子翻译成英文，注意不要损失任何信息\n{option}"
        eng_options = LLM_response(eng_options_prompt)
        # print("eng_options, ", eng_options)
    else:
        eng_context = context
        eng_ques = question
        eng_options = option
    
    simple_context = Logic_simple(eng_context)
    # print("simplified context: ", simple_context)
    logging.info(f"simplified context: {simple_context}")
    
    Lextract_context = Logic_Lextract(simple_context)
    # print("Lextracted context: ", Lextract_context)
    
    # xiaorong_prompt = "Please remove all natural language expressions from the following passage, and keep only the first-order logic expressions and their predicate definitions."
    # Lextract_context = LLM_response(xiaorong_prompt)
    
    logging.info(f"Lextracted context: {Lextract_context}")
    
    for i in range(4):
        # history_message = []
        if dataset == "logicqa":
            proposition = Logic_qwitho(question, option[i])
            eng_pro_prompt = f"请你将下面的题目翻译成英文，注意不要损失任何信息\n{proposition}"
            eng_proposition = LLM_response(eng_pro_prompt)
            # print("proposition: ", eng_proposition)
        else:
            eng_proposition = Logic_qwitho_eng(question, option[i])
            # print("proposition: ", eng_proposition)
            
        
        reasoning_process = Logic_deductive(Lextract_context, eng_proposition)
        
        # reasoning_process = Logic_deductive_noFOL(eng_context, eng_proposition)
        # print("Reasoning process: ", reasoning_process)
        logging.info(f"Reasoning process: {reasoning_process}")
        
        content_2 = f"{reasoning_process}Please carefully read the reasoning process above and summarize whether it considers the proposition to be valid or invalid. If invalid, output 'False' directly. If valid, output 'True' directly."
        response = LLM_response(content_2)
        logging.info(f"response to T/F: {response}")
        # print(response)
        if "true" in response.lower():
            answer.append(1)
            count += 1
            response_list.append(reasoning_process)
            if dataset == "logicqa":
                eng_option_prompt = f"请你将下面的句子翻译成英文，注意不要损失任何信息\n{option[i]}"
                eng_option = LLM_response(eng_option_prompt)
            else:
                eng_option = option[i]
            option_list.append(eng_option)
            # break
        else:
            answer.append(0)
        # # print(answer)
    # print(answer)
    logging.info(f"init answer: {answer}")
    if count == 1:
        for i in range(4):
            if answer[i] == 1:
                answer_0 = i
    elif count > 1:
        answer_0= Logic_m2one(eng_context, eng_ques, option_list, response_list, eng_options)
        # while True:
        #     try: 
        #         logging.info(f"Muti to One: {response_2}")
        #         # print("Muti to One:", response_2)
        #         content_4 = f"You now have an analysis: {response_2} and several options: {eng_options}. Please carefully read the analyse and directly output the number of the option that the analyse ultimately considers correct from the options, using letters A/B/C/D, Please direct output the A/B/C/D:"
        #         response_3 = LLM_response(content_4)
        #         answer_0 = int(extract_answer(response_3))
        #         break
        #     except:
        #         # print("try again")

        logging.info(f"final answer: {answer_0}")
        # print("final answer: ", answer_0)
    return answer, answer_0


def run(path, type_dataset, log):
    if type_dataset == "reclor":
        standard_answer = np.load(".\datasets\\reclor\\npy\\val.npy")
        # # print(type(option))
    else:
        standard_answer = np.load(".\datasets\logicqa\\npy\\text_eng.npy")
        # standard_answer = np.load(".\datasets2\logicqa\\npy\\arlsat_test.npy")
    dataset = load_dataset(path)
    answer = []
    answer_0 = []
    answer_direct_list = []
    count = 0
    direct_count = 0
    final_count = 0
    final_count_once = 0
    right_list = []
    logging.basicConfig(filename=log, level=logging.INFO, encoding='utf-8')
    dataset = dataset[count:]
    for sample in tqdm(dataset):
        # if start > count:
        #     count += 1
        #     answer = np.load(result).tolist()
        #     answer_0 = np.load(result2).tolist()
        #     continue
        if type_dataset == "reclor":
            # # print(sample)
            option = sample['answers']
            # # print(type(option))
        else:
            option = sample['options'].split('\n')
        context = sample['context']
        question = sample['question']
        
        # # for i in range(3):
        
        
        
        
        
        # # if type_dataset == "logicqa":
        # #     eng_con_prompt = f"请你将下面的题目翻译成英文，注意不要损失任何信息\n{context}"
        # #     eng_context = LLM_response(eng_con_prompt)
        # #     eng_ques_prompt = f"请你将下面的问题翻译成英文，注意不要损失任何信息\n{question}"
        # #     eng_ques = LLM_response(eng_ques_prompt)
        # #     eng_options_prompt = f"请你将下面的句子翻译成英文，注意不要损失任何信息\n{option}"
        # #     eng_options = LLM_response(eng_options_prompt)
        # #     # print("eng_options, ", eng_options)
        # # else:
        # #     eng_context = context
        # #     eng_ques = question
        # #     eng_options = option
        
        # # simple_context = Logic_simple(eng_context)
        # # # print("simplified context: ", simple_context)
        # # logging.info(f"simplified context: {simple_context}")
        
        # # Lextract_context = Logic_Lextract(simple_context)
        # # # print("Lextracted context: ", Lextract_context)
        # # logging.info(f"Lextracted context: {Lextract_context}")
        
        # # reasoning_process = Logic_nodeductive(Lextract_context)
        
        
        
        
        content = f"Context: {context}\nQuestion: {question}\nOptions:\n{option}\nDirectly output the symbol of the most suitable option A/B/C/D:"
        answer_res_direct = LLM_response(content)
        answer_direct = int(extract_answer(answer_res_direct))
        answer_direct_list.append(answer_direct)
        
        
        
        
            # if answer_direct == standard_answer[count]:
            #     break
        # print("answer_direct: ", answer_direct, " answer_standard: ", standard_answer[count])
        if answer_direct == standard_answer[count]:
            direct_count += 1
            # final_count += 1
            # print(f"Direct is already right! Direct right: {direct_count}, final right: {final_count}, total: {count+1}")
            logging.info(f"Direct is already right! Direct right: {direct_count}, final right: {final_count}, total: {count+1}")
            # answer_res = [0,0,0,0]
            # answer_res[answer_direct] = 1
            # answer_0_res = answer_direct
            
            
            
        # else:
        for i in range(3):
            answer_res, answer_0_res = Logic_Reasoner(context, question, option, type_dataset)
            if answer_0_res == standard_answer[count]:
                break
        if answer_0_res == standard_answer[count]:
            if i == 0:
                final_count_once += 1
            right_list.append(i)
            final_count += 1
            # print(f"Ours is finally right! Direct right: {direct_count}, final right: {final_count}, total: {count+1}")
            logging.info(f"Ours is finally right! Direct right: {direct_count}, final once right: {final_count_once},final right once: {final_count_once}, final right: {final_count}, total: {count+1}")
                # # print(answer_res)
                # np.append(answer, np.array(answer_res))
                
                
                
        answer.append(answer_res)
        answer_0.append(answer_0_res)
        # answer[count] = int(extract_answer(answer_res))
        # print(count)
        # print(answer)
        # print(answer[count])
        logging.info(f"count: {count}")
        logging.info(f"answer: {answer[count]}")
        logging.info(f"answers: {answer}")
        count += 1
        
        
        
        
        # np.save(result, np.array(answer))
        # np.save(result2, np.array(answer_0))
    logging.info(f"right list: {right_list}")
    logging.info(f"Direct right: {direct_count}, final once right: {final_count_once},final right once: {final_count_once}, final right: {final_count}, total: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dataset")
    
    parser.add_argument('--dataset', type=str, required=True, help="数据集的路径")
    parser.add_argument('--type', type=str, required=True, help="数据集名称")
    # parser.add_argument('--start', type=int, required=True, help="开始位置")
    # parser.add_argument('--result', type=str, required=True, help="存储位置")
    # parser.add_argument('--result2', type=str, required=True, help="另一个存储位置")
    parser.add_argument('--log', type=str, required=True, help="log")

    args = parser.parse_args()
    
    run(args.dataset, args.type, args.log)