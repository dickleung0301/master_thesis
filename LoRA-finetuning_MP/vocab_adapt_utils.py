import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HF_DATASETS_CACHE"] = "/export/data2/yleung/dataset"

import torch
import copy
import numpy as np
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# a IterableDataset to read the raw text line by line
class LineByLineDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, num_line, max_length=128, model_choice=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_choice = model_choice
        self.num_line = num_line

        # set padding token && padding side
        if model_choice in [1, 2]:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
    
    def __iter__(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()

                tokenized_inputs = self.tokenizer(
                    line,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                    return_tensors="pt"
                )

                labels = tokenized_inputs["input_ids"].clone()

                yield{
                    "input_ids": tokenized_inputs["input_ids"].squeeze(),
                    "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
                    "labels": labels.squeeze()
                }

    def __len__(self):
        return self.num_line

# a function to replace the tokenizer & word-embedd of the model
def vocab_adaptation(model, original_tokenizer, tokenizer_path, lora=False):

    # load the trained sentencepiece tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    # add the llama 3.1 Instruct special tokens
    original_tokenizer_special_tokens = {
        "bos_token": original_tokenizer.bos_token,
        "eos_token": original_tokenizer.eos_token,
        "unk_token": original_tokenizer.unk_token,
        "pad_token": original_tokenizer.pad_token,
        "additional_special_tokens": list(original_tokenizer.added_tokens_decoder.values()),
    }

    tokenizer.add_special_tokens({
        "bos_token": original_tokenizer_special_tokens["bos_token"],
        "eos_token": original_tokenizer_special_tokens["eos_token"],
        "additional_special_tokens": original_tokenizer_special_tokens["additional_special_tokens"]
    })

    # get the vocab size and model dimension of the custom tokenizer & the existing model
    vocab_size = len(tokenizer)
    model_dim = model.get_input_embeddings().embedding_dim
    model.config.vocab_size = vocab_size

    # get the overlapping vocabulary of the original tokenizer & the custom one
    original_vocab = original_tokenizer.get_vocab()
    custom_vocab = tokenizer.get_vocab()
    overlap_tokens = set(original_vocab.keys()).intersection(set(custom_vocab.keys()))

    # get the mapping of the overlapping tokens
    overlap_map = {token: (original_vocab[token], custom_vocab[token]) for token in overlap_tokens}

    # calculate the multivariate distribution of the word embeddings of llama
    original_embeddings = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)
    mean = np.mean(original_embeddings, axis=0)
    var = np.cov(original_embeddings, rowvar=False)
    #std = np.std(original_embeddings, axis=0)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(mean), torch.tensor(var))

    # replace the word embeddings & lm head
    new_embedding = torch.nn.Embedding(vocab_size, model_dim)
    #original_lm_head_weights = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)
    #lm_head_mean = np.mean(original_lm_head_weights, axis=0)
    #lm_head_var = np.cov(original_lm_head_weights, rowvar=False)
    #lm_head_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(lm_head_mean), torch.tensor(lm_head_var))
    new_lm_head = torch.nn.Linear(model_dim, vocab_size, bias=False)
    torch.nn.init.xavier_uniform_(new_lm_head.weight)

    # sample the new word embedd from the dist. of llama
    for i in range(vocab_size):
        with torch.no_grad():
            new_embedding.weight[i].copy_(dist.sample())
            #new_lm_head.weight[i].copy_(lm_head_dist.sample())
            #sampled_embedding = torch.normal(mean=torch.tensor(mean, dtype=torch.float32), std=torch.tensor(std, dtype=torch.float32))
            #new_embedding.weight[i].copy_(sampled_embedding)

    # to assign the overlapped word embedding
    for token, (idx1, idx2) in overlap_map.items():
        with torch.no_grad():
            new_embedding.weight[idx2].copy_(model.model.embed_tokens.weight[idx1])
            #new_lm_head.weight[idx2].copy_(model.lm_head.weight[idx1])

    # copy the first vocab_size word embedd from the original model
    if lora:
        trim_embeddings = model.model.embed_tokens.weight[:vocab_size].clone()
        with torch.no_grad():
            new_embedding.weight.copy_(trim_embeddings)

    model.set_input_embeddings(new_embedding)
    model.lm_head = new_lm_head

    # freeze the whole transformer body while only keeping word embeddings & lm head unfreezed
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the lm_head
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Unfreeze the embed_tokens
    if not lora:
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = True

    # inject lora trainable param to the word embedd
    if lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['embed_tokens'],
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'
        )

        print("####################\nlora config.\n####################")
        print(lora_config)

        # prepare the model for training with LoRA
        model = get_peft_model(model, lora_config)

    return model, tokenizer

def switch_llama_embedding(save_dir):
    
    # load the access token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')

    # login huggingface_hub
    login(token=token)

    # load the custom model & llama
    custom_model = AutoModelForCausalLM.from_pretrained(save_dir,
                                                        device_map='auto',
                                                        torch_dtype=torch.float16,
                                                        token=token)
    custom_tokenizer = AutoTokenizer.from_pretrained(save_dir, token=token)

    llama = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                                cache_dir='/export/data2/yleung/model_cache',
                                                device_map='cpu',
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                token=token)
    original_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', cache_dir="/export/data2/yleung/model_cache", token=token)

#    custom_model = AutoModelForCausalLM.from_pretrained(save_dir,
#                                                        device_map='cpu',
#                                                        torch_dtype=torch.float16,
#                                                        token=token)
#    custom_tokenizer = AutoTokenizer.from_pretrained(save_dir, token=token)
#
#    llama = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
#                                                cache_dir='/export/data2/yleung/model_cache',
#                                                device_map='auto',
#                                                torch_dtype=torch.float16,
#                                                low_cpu_mem_usage=True,
#                                                token=token)
#    original_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', cache_dir="/export/data2/yleung/model_cache", token=token)

    # get the device of word embeddings
    embedding_device = custom_model.get_input_embeddings().weight.device
#    embedding_device = llama.get_input_embeddings().weight.device

    # keep tracking the word embedds
    original_embed_tokens = copy.deepcopy(llama.model.embed_tokens).to(embedding_device)
#    original_embed_tokens = copy.deepcopy(custom_model.model.embed_tokens).to(embedding_device)

    # add the llama word embedding to the custom model
    custom_model.original_embed_tokens = original_embed_tokens
#    llama.original_embed_tokens = original_embed_tokens

    return original_tokenizer, custom_tokenizer, custom_model
