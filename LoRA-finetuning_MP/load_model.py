import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login

def model_factory(model_choice = 1):
    # load the access token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')

    # login huggingface_hub
    login(token=token)

    if model_choice == 1:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B',
                                                    cache_dir="/export/data2/yleung/model_cache",
                                                    device_map="cpu",
                                                    torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True,
                                                    token=token)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', cache_dir="/export/data2/yleung/model_cache", token=token)

        return model, tokenizer

    if model_choice == 2:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                                    cache_dir="/export/data2/yleung/model_cache",
                                                    device_map="cpu",
                                                    torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True,
                                                    token=token)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', cache_dir="/export/data2/yleung/model_cache", token=token)

        return model, tokenizer   