import json
from tqdm import tqdm
import re
import random
random.seed(0)
with open('/export/home/blyu/MetaTool/tool2question.json', 'r', encoding='utf-8') as f:
    tool2question = json.load(f)
    print(f"Load tool2question from /export/home/blyu/MetaTool/tool2question.json")

def extract_tool_des_list(prompt):
    # 提取出tool name:之后的，一直到第一个换行符之前的内容。包括tool name:和换行符
    pattern = re.compile(r'tool name: [^\n]*')
    tool_name = pattern.findall(prompt)
    assert len(tool_name) ==10, f"len(tool_name) != 10, {tool_name}"
    return tool_name

def create_new_demo(tool):
    # 分别创建关于该tool的3个正例和2个负例
    #tool name是tool name: 之后直到逗号之前的内容
    pattern = re.compile(r'tool name: [^,]*')
    tool_name = pattern.findall(tool)[0].split("tool name: ")[1]
    assert tool_name in tool2question, f"{tool_name} not in tool2question"
    tool=tool_name
    new_demo = []
    for i in range(3):
        new_query = random.choice(tool2question[tool])
        new_query = f"\"{new_query}\""
        new_demo.append(f"query: {new_query} judgement: Yes.")
    for i in range(2):
        new_tool = random.choice(list(tool2question.keys()))
        new_query = random.choice(tool2question[new_tool])
        new_query = f"\"{new_query}\""
        new_demo.append(f"query: {new_query} judgement: No.")
    return "\n".join(new_demo)
    

def convert2case1_level2(original_prompt,query):
    original_prifix = "Your current task is to choose the appropriate tool to solve the user's query based on their question. I will provide you with the user's question and information about the tools.\nIf there is a tool in the list that is applicable to this query, please return the name of the tool (you can only choose one tool). If there isn't, please return 'None.' Additionally, you will need to support your answer with a brief explanation."
    new_prifx = "Your current task is to judge whether the tool is appropriate to solve the user's query based on their question. I will provide you with the user's question and information about a tool.\nIf this is the tool that is applicable to this query, please return 'Yes.' If no, please return 'No.' Additionally, you will need to support your answer with a brief explanation."
    assert original_prifix in original_prompt, f"original_prifix not in original_prompt"
    new_prompt = original_prompt.replace(original_prifix, new_prifx)
    original="List of Tools with Names and Descriptions"
    new_= "Tool Name and Description"
    assert original in new_prompt, f"{original} not in new_prompt"
    new_prompt = new_prompt.replace(original, new_)
    tool_des_tag_start= "[Tool Name and Description Start]\n"
    tool_des_tag_end = "\n[Tool Name and Description End]"
    assert tool_des_tag_start in new_prompt, f"{tool_des_tag_start} not in new_prompt"
    assert tool_des_tag_end in new_prompt, f"{tool_des_tag_end} not in new_prompt"
    old_tool_des_part=new_prompt[new_prompt.find(tool_des_tag_start)+len(tool_des_tag_start):new_prompt.find(tool_des_tag_end)]
    tool_des_list=extract_tool_des_list(old_tool_des_part)
    #提取出[Examples Start]之后的内容
    assert "[Examples Start]" in new_prompt, f"[Examples Start] not in new_prompt"
    old_demo= new_prompt.split("[Examples Start]")[1]
    results=[]
    for tool in tool_des_list:
        new_demo=create_new_demo(tool)
        prompt=new_prompt.replace(old_tool_des_part, tool)
        prompt=prompt.replace(old_demo, new_demo) + "\n" + f"query: {query} judgement: "
        results.append(prompt)
    return results
original_data = json.load(open('/export/home/blyu/MetaTool-/metatool+.json', 'r'))

new_data = []
for i, item in enumerate(tqdm(original_data)):
    new_item = original_data[i]
    new_prompt = convert2case1_level2(item['action_prompt'], item['query'])
    new_item['onebyone_prompt'] = new_prompt
    new_data.append(new_item)

with open('/export/home/blyu/MetaTool-/metatool+.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(f"Save to /export/home/blyu/MetaTool-/metatool+.json")