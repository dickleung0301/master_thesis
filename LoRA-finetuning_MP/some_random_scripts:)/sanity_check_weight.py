import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM

# load the access token from .env
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

# login huggingface_hub
login(token=token)

model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                            cache_dir='/export/data2/yleung/model_cache',
                                            device_map='cpu',
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            token=token)

custom_model = AutoModelForCausalLM.from_pretrained('/export/data2/yleung/zht_32000',
                                            device_map='cpu',
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            token=token)

def sanity_check_frozen_body(model1, model2):
    # check the word embedd
    assert not torch.equal(model1.model.embed_tokens.weight, model2.model.embed_tokens.weight), f"The word embeddings are the same :("

    print("The word embeddings are the different :)")

    # check the lm head
    assert not torch.equal(model1.lm_head.weight, model2.lm_head.weight), f"The lm heads are the same :("

    print("The lm heads are the different :)")

    for (name1, param1), (name2, param2) in zip(model1.model.layers.named_parameters(), model2.model.layers.named_parameters()):
        assert name1 == name2, f"Parameter names differ: {name1} vs {name2}"

        assert param1.shape == param2.shape, f"Shape mismatch for {name1}: {param1.shape} vs {param2.shape}"

        assert torch.equal(param1, param2), f"Mismatch found in parameter: {name1}"

    print("All the params are equal")

sanity_check_frozen_body(model, custom_model)