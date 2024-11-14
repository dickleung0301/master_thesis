import os
from transformers import AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
login(token=token)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', cache_dir="/export/data2/yleung/model_cache", token=token)

special_tokens = {
    "bos_token": tokenizer.bos_token,
    "eos_token": tokenizer.eos_token,
    "unk_token": tokenizer.unk_token,
    "pad_token": tokenizer.pad_token,
    "additional_special_tokens": tokenizer.add_special_tokens,
}

print(special_tokens)