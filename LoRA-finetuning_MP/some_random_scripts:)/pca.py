import os
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import torch
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

def radius(x_vector, y_vector):

    # get the center of the cluster
    center_x = np.mean(x_vector)
    center_y = np.mean(y_vector)

    # calculate the distance of each point from the center
    distance = [np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in zip(x_vector, y_vector)]

    radius = max(distance)

    return center_x, center_y, radius

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

llama_embeddings = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)

custom_model = AutoModelForCausalLM.from_pretrained('/export/data2/yleung/zht_32000',
                                            device_map='cpu',
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            token=token)

custom_embeddings = custom_model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)

# apply pca to reduce to 2 dim
pca = PCA(n_components=2)
reduced_dim_llama = pca.fit_transform(llama_embeddings)
reduced_dim_custom = pca.fit_transform(custom_embeddings)

# calculate the radius
llama_center_x, llama_center_y, llama_radius = radius(reduced_dim_llama[:, 0], reduced_dim_llama[:, 1])
custom_center_x, custom_center_y, custom_radius = radius(reduced_dim_custom[:, 0], reduced_dim_custom[:, 1])

# plot the pca dist
plt.scatter(reduced_dim_llama[:, 0], reduced_dim_llama[:, 1], label="Llama 3.1 I 8B", marker="o", alpha=0.5)
plt.scatter(reduced_dim_custom[:, 0], reduced_dim_custom[:, 1], label="32000 Vocab Size", marker="x", alpha=0.5)

# circle the cluster
circle_llama = plt.Circle((llama_center_x, llama_center_y), llama_radius, color='blue', fill=False, linewidth=1.5, linestyle='--')
circle_custom = plt.Circle((custom_center_x, custom_center_y), custom_radius, color='red', fill=False, linewidth=1.5, linestyle='--')

# add circle to the plot
plt.gca().add_patch(circle_llama)
plt.gca().add_patch(circle_custom)

# labels
plt.title('PCA of Llama 3.1 I & 32000 Vocab Size Word Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# saving
plt.savefig('pca_we.png')
plt.close()