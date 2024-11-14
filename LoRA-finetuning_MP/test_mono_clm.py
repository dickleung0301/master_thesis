import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# load the access token from .env
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

# login huggingface_hub
login(token=token)

model = AutoModelForCausalLM.from_pretrained('/export/data2/yleung/zht_64000',
                                            device_map='auto',
                                            # cache_dir='/export/data2/yleung/model_cache',
                                            torch_dtype=torch.float16,
                                            token=token)
tokenizer = AutoTokenizer.from_pretrained('/export/data2/yleung/zht_64000', token=token)
first_device = next(model.parameters()).device

input = "我猜跟金銀角的寶"
tokenized_input = tokenizer(input, return_tensors='pt')
input_ids = tokenized_input['input_ids']
input_ids = input_ids.to(first_device)
generation = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)
generation_text = tokenizer.decode(generation[0], skip_special_tokens=True)
print(f"input text:{input} \ngenerated text: {generation_text}")

input = "這些措施，促進了蒙古"
tokenized_input = tokenizer(input, return_tensors='pt')
input_ids = tokenized_input['input_ids']
input_ids = input_ids.to(first_device)
generation = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)
generation_text = tokenizer.decode(generation[0], skip_special_tokens=True)
print(f"input text:{input} \ngenerated text: {generation_text}")

input = "唔知點解香港人個格"
tokenized_input = tokenizer(input, return_tensors='pt')
input_ids = tokenized_input['input_ids']
input_ids = input_ids.to(first_device)
generation = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)
generation_text = tokenizer.decode(generation[0], skip_special_tokens=True)
print(f"input text:{input} \ngenerated text: {generation_text}")

input = "袁弓夷冇耐之前都有提過美國"
tokenized_input = tokenizer(input, return_tensors='pt')
input_ids = tokenized_input['input_ids']
input_ids = input_ids.to(first_device)
generation = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)
generation_text = tokenizer.decode(generation[0], skip_special_tokens=True)
print(f"input text:{input} \ngenerated text: {generation_text}")
