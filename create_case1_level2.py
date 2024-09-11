import json
from tqdm import tqdm
import re
def conver_demo(original_prompt):
    # 提取出tool: 之后一直到换行符之前的内容
    pattern = re.compile(r'tool: [^\n]*')
    tool = pattern.findall(original_prompt)
    assert len(tool) ==5, f"len(tool) != 5"
    new_prompt = original_prompt
    for t in tool:
        if t == 'tool: None':
            new_prompt = new_prompt.replace(t, "judgment: No.")
        else:
            new_prompt = new_prompt.replace(t, "judgment: Yes.")
    return new_prompt
            

def convert2case1_level2(original_prompt):
    original_prifix = "Your current task is to choose the appropriate tool to solve the user's query based on their question. I will provide you with the user's question and information about the tools.\nIf there is a tool in the list that is applicable to this query, please return the name of the tool (you can only choose one tool). If there isn't, please return 'None.' Additionally, you will need to support your answer with a brief explanation."
    new_prifx = "Your current task is to judge whether the appropriate tool exists to solve the user's query based on their question. I will provide you with the user's question and information about the tools.\nIf there is a tool in the list that is applicable to this query, please return 'Yes.' If there isn't, please return 'No.' Additionally, you will need to support your answer with a brief explanation."
    assert original_prifix in original_prompt, f"original_prifix not in original_prompt"
    new_prompt = original_prompt.replace(original_prifix, new_prifx)
    #将最后的tool: 替换为 judgment:
    #只换最后一个
    new_prompt = new_prompt[:len(new_prompt)-len("tool: ")] + "judgment: "
    #提取出[Examples Start]之后的内容
    assert "[Examples Start]" in new_prompt, f"[Examples Start] not in new_prompt"
    demo= new_prompt.split("[Examples Start]")[1]
    demo = conver_demo(demo)
    new_prompt = new_prompt.split("[Examples Start]")[0] + "[Examples Start]" + demo
    return new_prompt

original_data = json.load(open('/export/home/blyu/MetaTool/dataset/tmp_dataset/Task2-Subtask3.json', 'r'))

new_data = []
for i, item in enumerate(tqdm(original_data)):
    new_item = original_data[i]
    new_prompt = convert2case1_level2(item['action_prompt'])
    new_item['binary_prompt'] = new_prompt
    new_data.append(new_item)

with open('/export/home/blyu/MetaTool/metatool+.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(f"Save to /export/home/blyu/MetaTool/metatool+.json")