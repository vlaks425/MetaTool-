import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json
data_file = "/home/2/uh02312/MetaTool-/metatool+.json"
output_file = "/home/2/uh02312/MetaTool-/metatool+p_output_70_bin_vllm.json"
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
# 如果输出文件已经存在,则不再运行
if os.path.exists(output_file):
    print(f"{output_file} already exists, skip")
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
else:
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id,trust_remote_code=True, \
        quantization="AWQ", max_num_seqs=4,max_model_len=4096,enable_prefix_caching=True)
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Load data from {data_file}")
    input_data = []
    for item in tqdm(data, desc="applying chat template",total=len(data)):
        input_data.append(tokenizer.apply_chat_template([{"role": "user", "content": item['binary_prompt+']}], tokenize=False))
    results = data
    sampling_params = SamplingParams(top_k=1,max_tokens=256)
    outputs = llm.generate(input_data,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    for i, output in enumerate(outputs):
        results[i]['binary_output+'] = output.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Save to {output_file}")

# 检查有多少个输出结果是以 Yes. 开头的
count = 0
invalid_count = 0
invalid_output=[]
for item in results:
    if item['binary_output+'].startswith("Yes."):
        count += 1
    elif item['binary_output+'].startswith("Judgment: "):
        if item['binary_output+'].split("Judgment: ")[1] == "Yes.":
            count += 1
    elif not item['binary_output+'].startswith("No."):
        invalid_count += 1
        invalid_output.append(item)
print(f"Total {count} outputs start with 'Yes.'")
print(f"Total {invalid_count} invalid outputs")
# 打印比例
print(f"Total {count/len(results)*100:.2f}% outputs start with 'Yes.'")
print(f"Total {invalid_count/len(results)*100:.2f}% invalid outputs")
with open(output_file.replace(".json","_invalid.json"), 'w', encoding='utf-8') as f:
    json.dump(invalid_output, f, indent=4, ensure_ascii=False)
    print(f"Save to {output_file.replace('.json','_invalid.json')}")
