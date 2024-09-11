import json
from tqdm import tqdm
import re
import random
random.seed(0)
with open('/export/home/blyu/MetaTool/tool2question.json', 'r', encoding='utf-8') as f:
    tool2question = json.load(f)
    print(f"Load tool2question from /export/home/blyu/MetaTool/tool2question.json")

with open('/export/home/blyu/MetaTool/dataset/plugin_des.json', 'r', encoding='utf-8') as f:
    tool_des = json.load(f)

def extract_tool_list(prompt):
    start="[List of Tools with Names and Descriptions Start]\n"
    end="[List of Tools with Names and Descriptions End]"
    assert start in prompt, f"{start} not in prompt"
    assert end in prompt, f"{end} not in prompt"
    tool_part=prompt[prompt.find(start)+len(start):prompt.find(end)]
    # 提取出tool name: 之后一直到逗号之前的内容
    pattern = re.compile(r'tool name: [^,]*')
    tool_name = pattern.findall(tool_part)
    assert len(tool_name) ==10, f"len(tool_name) != 10"
    tool_list=[]
    for t in tool_name:
        tool_list.append(t.split("tool name: ")[1])
    return tool_list


def extract_used_tool(prompt):
    demo_prompt = prompt.split("[Examples Start]\n")[1]
    tool_pattern = re.compile(r'tool: [^\n]*')
    tool = tool_pattern.findall(demo_prompt)[:-1]
    assert len(tool) ==5, f"len(tool) != 5, {tool}"
    return tool

def conver_tool_pool(prompt,gold_tool,replaced_tool):
    replaced_tool_des=tool_des[replaced_tool]
    gold_tool_des=tool_des[gold_tool]
    if "[\""+replaced_tool_des+"\"]" in prompt:
        replaced_tool_des="[\""+replaced_tool_des+"\"]"
        gold_tool_des="[\""+gold_tool_des+"\"]"
        replaced_seg=f"tool name: {replaced_tool}, tool description: {replaced_tool_des}"
        gold_seg=f"tool name: {gold_tool}, tool description: {gold_tool_des}"
    elif "['"+replaced_tool_des+"']" in prompt:
        replaced_tool_des="['"+replaced_tool_des+"']"
        gold_tool_des="['"+gold_tool_des+"']"
        replaced_seg=f"tool name: {replaced_tool}, tool description: {replaced_tool_des}"
        gold_seg=f"tool name: {gold_tool}, tool description: {gold_tool_des}"
    elif f"tool name: {replaced_tool}, tool description: {replaced_tool_des}" in prompt:
        # do nothing
        replaced_seg=f"tool name: {replaced_tool}, tool description: {replaced_tool_des}"
        gold_seg=f"tool name: {gold_tool}, tool description: {gold_tool_des}"
        pass
    else:
        #用正则表达匹配tool description
        #寻找以tool name: replaced_tool, tool description: 开头，到第一个换行符为止的内容
        pattern = re.compile(f'tool name: {replaced_tool}, tool description: [^\n]*')
        assert len(pattern.findall(prompt)) == 1, f"len(pattern.findall(prompt)) != 1"
        replaced_seg = pattern.findall(prompt)[0]
        gold_seg = f"tool name: {gold_tool}, tool description: {gold_tool_des}"
    assert replaced_seg in prompt, f"{replaced_seg} not in prompt"
    new_prompt=prompt.replace(replaced_seg,gold_seg)
    return new_prompt

def convert2case1_level2(action_prompt,binary_prompt,gold_tool):
    tool_list=extract_tool_list(action_prompt)
    used_tool=extract_used_tool(action_prompt)
    #  随机选取一个在tool_list但是不在used_tool中的tool
    replaced_tool=random.choice(list(set(tool_list)-set(used_tool)))
    new_binary_prompt=conver_tool_pool(binary_prompt,gold_tool,replaced_tool)
    return new_binary_prompt

original_data = json.load(open('/export/home/blyu/MetaTool/metatool+.json', 'r'))

new_data = []
for i, item in enumerate(tqdm(original_data)):
    new_item = original_data[i]
    new_prompt = convert2case1_level2(item['action_prompt'], item['binary_prompt'], item['tool'])
    new_item['binary_prompt+'] = new_prompt
    new_data.append(new_item)

with open('/export/home/blyu/MetaTool/metatool+.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(f"Save to /export/home/blyu/MetaTool/metatool+.json")