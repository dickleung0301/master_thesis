import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HF_DATASETS_CACHE"] = "/export/data2/yleung/dataset"
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
from load_data import *
from load_model import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from vocab_adapt_utils import * 
from torch.utils.data import DataLoader

# load the access token from .env
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

# login huggingface_hub
login(token=token)
#original_tokenizer, custom_tokenizer, custom_model = switch_llama_embedding('/export/data2/yleung/zht_32000')
custom_model = AutoModelForCausalLM.from_pretrained('/export/data2/yleung/zht_32000',
                                                    device_map='auto',
                                                    torch_dtype=torch.float16,
                                                    token=token)
custom_tokenizer = AutoTokenizer.from_pretrained('/export/data2/yleung/zht_32000', token=token)
print(custom_tokenizer.eos_token_id)
print(custom_tokenizer.eos_token)
print(custom_tokenizer.bos_token_id)
print(custom_tokenizer.bos_token)
first_device = next(custom_model.parameters()).device

input = "我猜跟金銀角的寶"
#original_tokenizer.padding_side = 'left'
#original_tokenizer.pad_token_id = original_tokenizer.eos_token_id
#custom_tokenizer.padding_side = 'left'
#custom_tokenizer.pad_token_id = custom_tokenizer.eos_token_id
tokenized_input = custom_tokenizer(input, return_tensors='pt')
input_ids = tokenized_input['input_ids']
#attention_mask = tokenized_input['attention_mask']
input_ids = input_ids.to(first_device)
#attention_mask = attention_mask.to(first_device)
#inputs_embeds = custom_model.original_embed_tokens(input_ids)
#inputs_embeds = inputs_embeds.to(first_device)
generation = custom_model.generate(input_ids=input_ids, max_new_tokens=50)
generation_text = custom_tokenizer.decode(generation[0], skip_special_tokens=True)
print(f"Testing generate: \ngenerated token ids: {generation[0]} \ngenerated text: {generation_text}")