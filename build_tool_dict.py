#从csv的第一列提取问题，第二列提取答案，生成答案到问题的字典

import csv
import json

csv_file = "/export/home/blyu/MetaTool/dataset/data/all_clean_data.csv"
results = {}
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    # Skip the header
    next(reader)
    for row in reader:
        if row[1] not in results:
            results[row[1]] = []
        results[row[1]].append(row[0])
with open("/export/home/blyu/MetaTool/tool2question.json", 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Save to /export/home/blyu/MetaTool/tool2question.json")