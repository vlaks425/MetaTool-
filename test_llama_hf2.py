import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16,"attn_implementation":"flash_attention_2"},
    device_map="cuda",
)

messages = [
    {"role": "user", "content": "\"You are a helpful AI assistant. Your current task is to judge whether the appropriate tool exists to solve the user's query based on their question. I will provide you with the user's question and information about the tools.\nIf there is a tool in the list that is applicable to this query, please return 'Yes.' If there isn't, please return 'No.' Additionally, you will need to support your answer with a brief explanation.\nUser's Query:\n[User's Query Start]\nHow can I make this experience as immersive as possible?\n[User's Query End].\nList of Tools with Names and Descriptions:\n[List of Tools with Names and Descriptions Start]\n1. tool name: PolishTool, tool description: Elevate your content with our AI-powered tool, which utilizes advanced rewriting techniques to create more human-like expressions and foster creative inspiration.\n2. tool name: locator, tool description: ['Add-on for displaying the current coordinates of the ISS and the names of the current astronauts in space.']\n3. tool name: ai_council, tool description: ['The AI council assesses queries through various agents, offering insights from many perspectives.']\n4. tool name: DataRetrievalTool, tool description: An tool that expands memory. It stores and retrieves user information to provide personalized assistance and generates real-time chat responses using the knowledge from the document collection.\n5. tool name: Zapier, tool description: ['Interact with over 5,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and thousands more.']\n6. tool name: tailor_erp, tool description: ['A tool to help creating tailor-made ERP application.']\n7. tool name: recipe_retrieval, tool description: [\"Discover recipe ideas, meal plans and cooking tips from Tasty's millions of users!\"]\n8. tool name: web_scraper, tool description: ['Scrape content from webpages by providing a URL.', 'Scrape content from webpages by providing a URL.']\n9. tool name: SSH, tool description: ['Ability to SSH into your server and turn your natural language into server commands. ']\n10. tool name: WordCloud, tool description: ['Create word cloud images from text.']\n\n[List of Tools with Names and Descriptions End]\"\nHere are some examples:\n[Examples Start]\nquery: \"I'm looking to send out personalized emails to a list of recipients in Mailchimp. Can you show me how to do that?\" tool: Zapier\nquery: \"Could you generate a word cloud image for this speech transcript I have?\" tool: WordCloud\nquery: \"Hey, could you please assist me by checking if there are any available car park lots at the HDB car park located in Bukit Merah View?\" tool: None\nquery: \"What organizations are currently conducting research on Alzheimer's disease?\" tool: None\nquery: \"I want to rephrase my cover letter to make it more compelling and unique, can you help me with that?\" tool: PolishTool\nUser query: \"How can I make this experience as immersive as possible?\" judgment: "},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    do_sample=False,
)
print(outputs[0]["generated_text"][-1]['content'])

