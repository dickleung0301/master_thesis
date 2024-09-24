import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM
from huggingface_hub import login
import torch

# load the model
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
login(token=token)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                            cache_dir="/export/data2/yleung/model_cache",
                                            device_map="cpu",
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            token=token)

# state the vocab size and get the model dimensions
vocab_size = 32000
model_dim = model.get_input_embeddings().embedding_dim
new_embedding = torch.nn.Embedding(vocab_size, model_dim)

# initialise the embedding weight
torch.nn.init.normal_(new_embedding.weight, mean=0.0, std=model.config.initializer_range)

# replace the embedding 
model.set_input_embeddings(new_embedding)

print(model)