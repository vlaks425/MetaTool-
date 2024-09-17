import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ['CURL_CA_BUNDLE'] = ''
#os.environ['REQUESTS_CA_BUNDLE'] = ''
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json
data_file = "/export/home/blyu/MetaTool-/metatool+.json"
output_file = "/export/home/blyu/MetaTool-/metatool+_output_1by1_vllm.json"
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# 如果输出文件已经存在,则不再运行
if os.path.exists(output_file):
    print(f"{output_file} already exists, skip")
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
else:
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id,quantization=None,tensor_parallel_size=1,
        dtype="bfloat16",max_num_seqs=1)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = data[:10]
    print(f"Load data from {data_file}")
    input_data = []
    for item in tqdm(data, desc="applying chat template",total=len(data)):
        for prompt in item['onebyone_prompt']:
            input_data.append(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False))
    results = data
    sampling_params = SamplingParams(max_tokens=256,top_k=1)
    outputs = llm.generate(input_data,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Save to {output_file}")

error_count = 0
invalid_count = 0
invalid_output=[]
for i in range(0,len(results),10):
    for j in range(10):
        anwser = results[i+j]['onebyone_prompt'][j].split("\n")[0]
        if "Yes" in anwser:
            error_count += 1
            break
        elif "No" in anwser:
            continue
        else:
            invalid_count += 1
            invalid_output.append(results[i+j])
            break
print(f"Total {error_count} outputs start with 'Yes.'")
print(f"Total {invalid_count} invalid outputs")
#打印比例
print(f"Total {error_count/len(results)*100:.2f}% outputs start with 'Yes.'")
print(f"Total {invalid_count/len(results)*100:.2f}% invalid outputs")
with open(output_file.replace(".json","_invalid.json"), 'w', encoding='utf-8') as f:
    json.dump(invalid_output, f, indent=4, ensure_ascii=False)
    print(f"Save to {output_file.replace('.json','_invalid.json')}")
