import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import torch
from dotenv import load_dotenv
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from exception import *

def model_factory(model_name):
    try:
        # load the hugging face credentials from .env file
        load_dotenv()
        token = os.getenv("HUGGINGFACE_TOKEN")
        login(token=token)

        if model_name == "t5-small" or model_name == "t5-base" or model_name == "t5-large":
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_name, load_in_4bit=True)
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
        elif model_name == "meta-llama/Llama-2-7b-chat-hf" or model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_name == "meta-llama/Meta-Llama-3.1-8B":
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        cache_dir="/export/data2/yleung/model_cache",
                                                        device_map='auto',
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True,
                                                        token=token)
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    cache_dir="/export/data2/yleung/model_cache",
                                                    token=token)

        return model, tokenizer

    except Exception as e:
        raise ModelInitializationError(f"There is somethin wrong in model factory: {e}")