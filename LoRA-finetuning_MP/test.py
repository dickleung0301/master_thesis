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
from utils import *

# load the access token from .env
#load_dotenv()
#token = os.getenv('HUGGINGFACE_TOKEN')

# login huggingface_hub
#login(token=token)

#model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                            #cache_dir='/export/data2/yleung/model_cache',
                                            #device_map="auto",
                                            #torch_dtype=torch.float16,
                                            #token=token)
#tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', cache_dir="/export/data2/yleung/model_cache", token=token)
#first_device = next(model.parameters()).device

#input = "Hello World!"
#tokenizer.padding_side = 'left'
#tokenizer.pad_token_id = tokenizer.eos_token_id
#print(f"input: {input}")
#tokenized_input = tokenizer(input, return_tensors='pt', return_attention_mask=True)
#input_ids = tokenized_input['input_ids']
#attention_mask = tokenized_input['attention_mask']
#print(f"input_ids: {input_ids}")
#input_ids.to(first_device)
#attention_mask.to(first_device)
#inputs_embeds = model.model.embed_tokens(input_ids)
#print(f"inputs_embeds: {inputs_embeds}")
#inputs_embeds.to(first_device)
#generation_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)
#generation_text_ids = tokenizer.decode(generation_ids[0])
#print(f"Testing generate: \ngenerated token ids: {generation_ids[0]} \ngenerated text: {generation_text_ids}")
#generation = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0)
#generation_text = tokenizer.decode(generation[0])
#print(f"Testing generate: \ngenerated token ids: {generation[0]} \ngenerated text: {generation_text}")
#outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
#print(f"logits shape: {outputs.logits.to('cpu').shape}")
#print(f"logits: {outputs.logits.to('cpu')}")
#print(outputs.past_key_values)
#generated_token = tokenizer.decode(torch.argmax(outputs.logits.to('cpu'), dim=-1)[0][-1].item())
#print(f"ouput: {generated_token}")
#generated_token = tokenizer.decode(torch.argmax(outputs.logits.to('cpu'), dim=-1)[0][-2].item())
#print(f"last token of input: {generated_token}")
#outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
#check_tokenizer(tokenizer_path='/export/data2/yleung/zht_32000', corpus_path='/export/data2/yleung/smp_zht/train_sets/trainset_1/1.txt')
vocab_len_stat(tokenizer_path='/export/data2/yleung/zht_32000')