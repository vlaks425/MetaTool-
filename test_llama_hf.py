import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig


model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config,attn_implementation="flash_attention_2")

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=True))
