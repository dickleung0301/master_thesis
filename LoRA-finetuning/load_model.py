import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login

def load_llama_3():
    # load the access token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')
    
    # login huggingface_hub
    login(token=token)

    # load the model & tokenizer
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B', load_in_4bit=True, token=token)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', token=token)

    return model, tokenizer

def model_factory(model_choice = 1):

    match model_choice:
        case 1:
            model, tokenizer = load_llama_3()
            return model, tokenizer