import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import re
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


data_file = "/export/home/blyu/MetaTool/metatool+.json"
output_file = "/export/home/blyu/MetaTool/metatool+_output_70b_gaming_vllm.json"
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
if os.path.exists(output_file):
    print(f"{output_file} already exists, skip")
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
else:
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id,quantization=None,tensor_parallel_size=4,
        dtype="bfloat16",max_num_seqs=4)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Load data from {data_file}")
    input_data = []
    for item in tqdm(data, desc="applying chat template",total=len(data)):
        input_data.append(tokenizer.apply_chat_template([{"role": "user", "content": item['gaming_prompt']}], tokenize=False))
    results = data
    sampling_params = SamplingParams(top_k=1,max_tokens=256)
    outputs = llm.generate(input_data,
                   sampling_params=sampling_params,
                   use_tqdm=True)
    for i, output in enumerate(outputs):
        results[i]['gaming_output'] = output.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Save to {output_file}")
assert len(results) == len(data), f"len(results) != len(data)"
error_count = 0
invalid_count = 0
invalid_output=[]
for i in range(len(results)):
    output=results[i]['gaming_output']
    input_prompt = results[i]['gaming_prompt']
    tool_list=extract_tool_list(input_prompt)
    # 提取输出中直到第一个换行符为止的内容
    model_judgment = output.split("\n")[0]
    if "tool: " in model_judgment:
        model_judgment = model_judgment.split("tool: ")[1]
    if "tool name: " in model_judgment:
        model_judgment = model_judgment.split("tool name: ")[1]
    if "Tool: " in model_judgment:
        model_judgment = model_judgment.split("Tool: ")[1]
    #如果是句号结尾，去掉句号
    if model_judgment.endswith("."):
        model_judgment = model_judgment[:-1]
    if model_judgment.endswith(" "):
        model_judgment = model_judgment[:-1]
    if model_judgment in tool_list:
        error_count += 1
    else:
        invalid_count += 1
        invalid_output.append(results[i])
#打印总数
print(f"Total {len(results)} outputs.")
print(f"Total {error_count} outputs start with a tool name.")
#打印比例
print(f"Total {invalid_count} outputs do not start with a tool name.")
with open(output_file.replace(".json","_invalid.json"), 'w', encoding='utf-8') as f:
    json.dump(invalid_output, f, indent=4, ensure_ascii=False)
    print(f"Save to {output_file.replace('.json','_invalid.json')}")

