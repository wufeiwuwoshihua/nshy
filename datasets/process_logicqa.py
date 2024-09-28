import json
import numpy as np
import argparse
import os


def parse_line(line):
    return line.strip()


def get_answer(correct_choice):
    answer = ord(correct_choice) - 97
    return answer


def txt_to_json_and_npy(txt_file, json_file, npy_file):
    try:
        with open(txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 按照空行分割文本
        samples = []
        current_sample = []

        for line in lines:
            if line.strip() == "":
                if current_sample:
                    samples.append(current_sample)
                    current_sample = []
            else:
                current_sample.append(line)
        
        # 添加最后一个样例
        if current_sample:
            samples.append(current_sample)
        
        # 处理每个样例并转换为JSON对象
        if os.path.exists(json_file):
            print("文件已存在，附加")
            with open(json_file, 'r', encoding='utf-8') as file:
                data_list = json.load(file)
        else:
            data_list = []
        
        answers = []
        
        for sample in samples:
            # print(sample)
            if len(sample) < 7:
                raise ValueError("某个样例的内容不足，至少需要8行")
            
            answer = get_answer(parse_line(sample[0]))
            context = parse_line(sample[1])
            question = parse_line(sample[2])
            options = "\n".join([parse_line(line) for line in sample[3:7]])

            data = {
                "context": context,
                "question": question,
                "options": options,
                "answer": answer,
            }
            
            data_list.append(data)
            answers.append(answer)
        
        # 将所有JSON对象写入文件
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(data_list, file, ensure_ascii=False, indent=4)
        
        
        # 将答案数组保存为 .npy 文件
        if os.path.exists(npy_file):
            existing_answers = np.load(npy_file)
            all_answers = np.concatenate((existing_answers, np.array(answers)))
        else:
            all_answers = np.array(answers)
        np.save(npy_file, all_answers)
        
        
        print(f'转换成功！JSON文件已保存为 {json_file}')
        print(f'答案已保存为 {npy_file}')
    
    except Exception as e:
        print(f'发生错误: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将包含多个样例的TXT文件转换为JSON文件，并将答案保存为 .npy 文件')
    parser.add_argument('--txt_file', help='要转换的TXT文件路径')
    parser.add_argument('--json_file', help='转换后保存的JSON文件路径')
    parser.add_argument('--npy_file', help='转换后保存的.npy文件路径')
    
    args = parser.parse_args()
    
    txt_to_json_and_npy(args.txt_file, args.json_file, args.npy_file)
