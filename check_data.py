import json
data_file="/export/home/blyu/MetaTool/metatool+.json"
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Load data from {data_file}")
print(f"Total data: {len(data)}")

#print(data[0]["binary_prompt"])
#打印[Examples Start]之后的内容
print(data[300]["action_prompt"].split("[Examples Start]")[1])