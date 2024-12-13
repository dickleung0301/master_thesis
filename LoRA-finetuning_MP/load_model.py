import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["HF_DATASETS_CACHE"] = "/export/data2/yleung/dataset"

import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login

def model_factory(model_choice=1, device_map='cpu'):
    # load the access token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')

    # login huggingface_hub
    login(token=token)

    if model_choice == 1:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B',
                                                    cache_dir='/export/data2/yleung/model_cache',
                                                    device_map=device_map,
                                                    torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True,
                                                    token=token)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', cache_dir="/export/data2/yleung/model_cache", token=token)

        return model, tokenizer

    if model_choice == 2:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                                    cache_dir='/export/data2/yleung/model_cache',
                                                    device_map=device_map,
                                                    torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True,
                                                    token=token)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', cache_dir="/export/data2/yleung/model_cache", token=token)

        return model, tokenizer   
    
    if model_choice == 3:
        model = AutoModelForCausalLM.from_pretrained('Unbabel/TowerInstruct-7B-v0.2',
                                                    cache_dir='/export/data2/yleung/model_cache',
                                                    device_map=device_map,
                                                    torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True,                                          
                                                    token=token)
        tokenizer = AutoTokenizer.from_pretrained('Unbabel/TowerInstruct-7B-v0.2', cache_dir="/export/data2/yleung/model_cache", token=token)

        return model, tokenizer

def load_embed_tokens_and_tokenizer(model_choice):
    # load the access token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')

    # login with huggingface_hub
    login(token=token)

    # load the embed_tokens & tokenizer
    model, tokenizer = model_factory(model_choice=model_choice)

    # do a deep copy for the embed_tokens
    try:
        encoding_embedd = copy.deepcopy(model.model.embed_tokens)
    except ValueError as e:
        print("Caught an exception:", e)
        raise

    return encoding_embedd, tokenizer