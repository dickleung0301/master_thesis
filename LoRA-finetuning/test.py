import os
import torch
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
dataset = load_alma('train', 'de-en')
dataset = finetuning_preprocess(dataset, 'translation', 'de', 'en', tokenizer)
print('input_ids')
print(dataset[0]['input_ids'])
print('attention_mask')
print(dataset[0]['attention_mask'])
print('labels')
print(dataset[0]['labels'])
#inputs_ids, labels_ids = preprocess(dataset, 'translation', 'zh', 'en', tokenizer)
#dataloader = create_dataloader(dataset=dataset, key='translation', src_lang='zh', trg_lang='en', tokenizer=tokenizer, device=device)
print(tokenizer('<|eot_id|>'))