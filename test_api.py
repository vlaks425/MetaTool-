import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CURL_CA_BUNDLE'] = '/export/home/blyu/ZscalerRootCertificate.crt'
#os.environ['REQUESTS_CA_BUNDLE'] = ''
import requests
import json

def send_api_request(message):
    url = "http://hibari1.svp.cl.nec.co.jp:10002/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [{"role": "user", "content": message}],
        "max_tokens": 32,
        "temperature": 0.3,
        "stream": False,
        "model": "llama3.1-70b-awq"
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"エラー: {response.status_code}, {response.text}"

# スクリプトの使用例
if __name__ == "__main__":
    user_message = "hello"
    result = send_api_request(user_message)
    print(json.dumps(result, indent=2, ensure_ascii=False))