import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import torch
data_file = "/export/home/blyu/MetaTool/metatool+.json"
output_file = "/export/home/blyu/MetaTool/metatool+_output_70b_orig_vllm.json"
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
if os.path.exists(output_file):
    print(f"{output_file} already exists, skip")
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
else:
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id,tensor_parallel_size=4,trust_remote_code=True, \
        quantization="AWQ", max_num_seqs=4)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Load data from {data_file}")
    input_data = []
    for item in tqdm(data, desc="applying chat template",total=len(data)):
        input_data.append(tokenizer.apply_chat_template([{"role": "user", "content": item['action_prompt']}], tokenize=False))
    results = data
    sampling_params = SamplingParams(top_k=1,max_tokens=256)
    outputs = llm.generate(input_data,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    for i, output in enumerate(outputs):
        results[i]['orig_output'] = output.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Save to {output_file}")

#检查有多少个输出结果是以 None. 开头的
count = 0
for item in results:
    if item['orig_output'].startswith("None"):
        count += 1
print(f"Total {count} outputs start with 'None.'")
#打印比例
print(f"Total {count/len(results)*100:.2f}% outputs start with 'None.'")

