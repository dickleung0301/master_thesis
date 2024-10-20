import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
login(token=token)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=token)
tokenizer.pad_token_id = tokenizer.eos_token_id

example = "testing for bos token :)"
encoded_example = tokenizer.encode(example, padding='max_length', max_length=30)
decoded_example = tokenizer.decode(encoded_example, skip_special_tokens=False)
print("Original Text:")
print(example)
print("Encoded Tokens:")
print(encoded_example)
print("Decoded Tokens:")
print(decoded_example)