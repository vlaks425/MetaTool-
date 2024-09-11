import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import transformers
import torch
from tqdm import tqdm
import json
data_file = "/export/home/blyu/MetaTool/metatool+.json"
output_file = "/export/home/blyu/MetaTool/metatool+_output_bin.json"
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16,"attn_implementation":"flash_attention_2"},
    device_map="cuda",
)
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Load data from {data_file}")

results = data
for i, item in enumerate(tqdm(data)):
    messages = [
        {"role": "user", "content": item['binary_prompt']},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        do_sample=False,
    )
    results[i]['binary_output'] = outputs[0]["generated_text"][-1]['content']
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Save to {output_file}")

#检查有多少个输出结果是以 No. 开头的
count = 0
invalid_count = 0
for item in results:
    if item['binary_output'].startswith("No."):
        count += 1
    # 计算非法输出,即不是以Yes. 或 No. 开头的
    if not item['binary_output'].startswith("Yes.") and not item['binary_output'].startswith("No."):
        invalid_count += 1
print(f"Total {count} outputs start with 'No.'")
print(f"Total {invalid_count} invalid outputs")
#打印比例
print(f"Total {count/len(results)*100:.2f}% outputs start with 'No.'")
print(f"Total {invalid_count/len(results)*100:.2f}% invalid outputs")

