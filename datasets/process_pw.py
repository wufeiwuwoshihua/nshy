import json
import random
import numpy as np

def jsonl_to_json(jsonl_file_path, json_file_path, npy_file_path):
    # 用来存储转换后的数据
    data = []
    answers = []

    # 读取jsonl文件
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            # print('# 解析每一行的JSON对象')
            entry = json.loads(line.strip())
            questions = entry.get('questions', [])

            if questions:
                # print('# 随机选择一个问题')
                question_id = random.choice(list(questions.keys()))
                selected_question = questions[question_id]
                # 获取选中问题的ID
                # question_id = selected_question.get('id', '')
                # 获取选中问题的文本
                question_text = selected_question.get('question','')
                # 获取选中问题的标签作为答案
                question_label = selected_question.get('answer')
                # print(question_label)
                if question_label == True:
                    answer = 1
                else:
                    answer = 0
            

                filtered_entry = {
                    'qid':question_id,
                    'context': entry.get('theory', ''),
                    'inference': question_text,
                    'answer':answer  
                }
            else:
                continue
            # 将每个条目添加到列表中
            data.append(filtered_entry)
            answers.append(answer)
    
    # 将数据写入到JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    np.save(npy_file_path, answers)


# 使用示例
if __name__ == '__main__':
    jsonl_file_path = r'C:\Users\24838\Desktop\datasets\pw\meta-test.jsonl'  # 输入的JSONL文件路径
    json_file_path = r'pw\json\val.json'   # 输出的JSON文件路径
    npy_file_path = r'pw\npy\val.npy'
    jsonl_to_json(jsonl_file_path, json_file_path, npy_file_path)

