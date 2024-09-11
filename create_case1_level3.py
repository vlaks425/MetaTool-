import json
from tqdm import tqdm
import re
import random
random.seed(0)
with open('/export/home/blyu/MetaTool/tool2question.json', 'r', encoding='utf-8') as f:
    tool2question = json.load(f)
    print(f"Load tool2question from /export/home/blyu/MetaTool/tool2question.json")

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


def conver_demo(original_prompt, tool_list):
    # 提取出tool: 之后一直到换行符之前的内容
    tool_pattern = re.compile(r'tool: [^\n]*')
    tool = tool_pattern.findall(original_prompt)[:-1]
    assert len(tool) ==5, f"len(tool) != 5, {tool}"
    # 提取出query: 之后一直到tool: 为止的内容，但是不包括tool:
    query_pattern = re.compile(r'query:\s*(.*?)\s*tool:')
    query = query_pattern.findall(original_prompt)[:-1]
    assert len(query) ==5, f"len(query) != 5, {query}"
    new_prompt = original_prompt
    for i in range(5):
        if tool[i] == 'tool: None':
            seg=query[i]+" "+tool[i]
            assert seg in new_prompt, f"{seg} not in new_prompt"
            #从tool_list中随机选择一个tool
            new_tool = random.choice(tool_list)
            #从tool2question中随机选择一个question
            new_query = random.choice(tool2question[new_tool])
            new_seq = f"query: {new_query} tool: {new_tool}"
            new_prompt = new_prompt.replace(seg, new_seq)
    return new_prompt
            

def convert2case1_level2(original_prompt):
    tool_list=extract_tool_list(original_prompt)
    original_prifix = "Your current task is to choose the appropriate tool to solve the user's query based on their question. I will provide you with the user's question and information about the tools.\nIf there is a tool in the list that is applicable to this query, please return the name of the tool (you can only choose one tool). If there isn't, please return 'None.' Additionally, you will need to support your answer with a brief explanation."
    new_prifx = "Your current task is to choose the appropriate tool to solve the user's query based on their question. I will provide you with the user's question and information about the tools.\nPlease return the name of the tool (you can only choose one tool). Additionally, you will need to support your answer with a brief explanation."
    assert original_prifix in original_prompt, f"original_prifix not in original_prompt"
    new_prompt = original_prompt.replace(original_prifix, new_prifx)
    #提取出[Examples Start]之后的内容
    assert "[Examples Start]" in new_prompt, f"[Examples Start] not in new_prompt"
    demo= new_prompt.split("[Examples Start]")[1]
    demo = conver_demo(demo, tool_list)
    new_prompt = new_prompt.split("[Examples Start]")[0] + "[Examples Start]" + demo
    return new_prompt

original_data = json.load(open('/export/home/blyu/MetaTool/metatool+.json', 'r'))

new_data = []
for i, item in enumerate(tqdm(original_data)):
    new_item = original_data[i]
    new_prompt = convert2case1_level2(item['action_prompt'])
    new_item['gaming_prompt'] = new_prompt
    new_data.append(new_item)

with open('/export/home/blyu/MetaTool/metatool+.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(f"Save to /export/home/blyu/MetaTool/metatool+.json")