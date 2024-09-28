import json
import numpy as np
def jsonl_to_json(jsonl_file_path, json_file_path, npy_file_path):
    # 读取jsonl文件
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        data = []
        answers = []
        for line in jsonl_file:
            # 解析jsonl格式的每一行
            entry = json.loads(line.strip())
            # 只保留需要的字段
            premises = "\n".join(entry.get("premises", []))

            simplified_entry = {
                'context': premises,
                #'premises-FOL': entry.get('premises-FOL', []),
                'inference': entry.get('conclusion', ''),
                #'conclusion-FOL': entry.get('conclusion-FOL', ''),
                #'label': entry.get('label', '')
                'answer': entry.get('label', '')
            }
            if simplified_entry['answer'] == 'Uncertain':
                continue
            elif simplified_entry['answer'] == 'True':
                simplified_entry['answer'] = 1
            else:
                simplified_entry['answer'] = 0
            data.append(simplified_entry)
            answers.append(simplified_entry['answer'])
    
    # 将数据写入到JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
        
    np.save(npy_file_path, answers)

# 使用示例
if __name__ == '__main__':
    jsonl_file_path = r'C:\Users\24838\Desktop\datasets\FOLIO\folio-validation.jsonl'  # 输入的JSONL文件路径
    json_file_path = r'FOLIO\json\val.json'   # 输出的JSON文件路径
    npy_file_path = r'FOLIO\npy\val.npy'
    jsonl_to_json(jsonl_file_path, json_file_path, npy_file_path)
