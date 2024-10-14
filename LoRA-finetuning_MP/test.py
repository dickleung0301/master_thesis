import os
import torch
import bitsandbytes as bnb
from dotenv import load_dotenv
from huggingface_hub import login
from load_data import *
from transformers import AutoTokenizer

load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
login(token=token)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=token)
# load env var
#load_dotenv()
#token = os.getenv('HUGGINGFACE_TOKEN')

# login
#login(token=token)

# device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load llama tokenizer
#tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=token)

# test create dataloader
dataset = load_alma('train', 'zh-en')
processed_dataset = finetuning_preprocess(dataset, 'translation', 'zh', 'en', 'zh-en', tokenizer, True)
print(tokenizer.encode('<|end_header_id|>'))
print(tokenizer.encode('<|eot_id|>'))
print(processed_dataset.input_ids[0])
print(processed_dataset.attention_mask[0])
print(processed_dataset.labels[0])
# inputs_ids, labels_ids = preprocess(dataset, 'translation', 'zh', 'en', tokenizer)
# dataloader = create_dataloader(dataset=dataset, key='translation', src_lang='zh', trg_lang='en', tokenizer=tokenizer, device=device)
# print(tokenizer.eos_token)
# print(dir(bnb))
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())