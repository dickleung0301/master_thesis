import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from huggingface_hub import login

def freeze_trans_body(model):
    # load the access token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')

    # login huggingface_hub
    login(token=token)

    full_precision_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                                                cache_dir='/export/data2/yleung/model_cache',
                                                                device_map='cpu',
                                                                torch_dtype=torch.float32,
                                                                token=token)

    # replace the fp16 word embedds & lm head with fp32
    model.set_input_embeddings(full_precision_model.model.embed_tokens)
    model.lm_head = full_precision_model.lm_head

    # freeze the transformer body
    for name, param in model.named_parameters():
        if "lm_head" in name or "embed_tokens" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def check_tokenizer(tokenizer_path, corpus_path, batch_size=128):

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # to store the original line, token ids, and the decoded line
    original_line = []
    token_ids = []
    decoded_line = []

    with open(corpus_path, 'r') as file:
        for _ in range(batch_size):
            line = file.readline().strip()
            original_line.append(line)
            temp = tokenizer.encode(line)
            token_ids.append(temp)
            decoded_line.append(tokenizer.decode(temp, skip_special_tokens=True))

    output = {
        'original_line': original_line,
        'token_ids': token_ids,
        'decoded_line': decoded_line
    }
    df = pd.DataFrame(output)
    df.to_csv('check_tokenizer.csv', index=False)

def vocab_len_stat(tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab = tokenizer.get_vocab()

    tokens_length = [len(token) for token in vocab.keys()]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(tokens_length, bins=range(1, max(tokens_length) + 2), edgecolor='black', align='left')
    plt.xlabel("Token Length (in Chinese characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Token Lengths")
    plt.xticks(range(1, max(tokens_length) + 1))

    plt.savefig("token_length_distribution.png", format="png", dpi=300)

def norm_test(lm_head, vocab_size):

    # print the shape of the lm_head
    lm_head_weight = lm_head.weight.detach()
    print(lm_head_weight.shape)

    # calculate the euclidean norm for each hidden dim
    global_norm = torch.norm(lm_head_weight, p=2).item()

    # average the norm
    average_norm = global_norm / (vocab_size ** 0.5)

    return average_norm

def generation_for_single_input(model, tokenizer, input, do_sample, temperature, top_p, repetition_penalty):
    
    # get the device of the model
    first_device = next(model.parameters()).device

    # encode the input
    tokenized_input = tokenizer(input, return_tensors='pt', return_attention_mask=True)

    # get the input_ids & attention_mask
    input_ids = tokenized_input['input_ids']
    input_ids = input_ids.to(first_device)
    attention_mask = tokenized_input['attention_mask']
    attention_mask = attention_mask.to(first_device)

    # generation & decoding
    generation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=250, do_sample=do_sample, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    generated_text = tokenizer.decode(generation[0], skip_special_tokens=True)
    print(f"input text: {input}\ngenerated text: {generated_text}")

def generation_for_single_input_vocab_adapt(model, encoder, decoder, input, do_sample, temperature, top_p, repetition_penalty):

    # get the device of the model
    first_device = next(model.parameters()).device

    # encode the input
    tokenized_input = encoder(input, return_tensors='pt', return_attention_mask=True)

    # get the input_ids & attention_mask
    input_ids = tokenized_input['input_ids']
    input_ids = input_ids.to(first_device)
    attention_mask = tokenized_input['attention_mask']
    attention_mask = attention_mask.to(first_device)

    # encode the input id to input embedd by original word embedd
    inputs_embeds = model.original_embed_tokens(input_ids)

    # generation & decoding
    generation = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=50, do_sample=do_sample, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    generated_text = decoder.decode(generation[0], skip_special_tokens=True)
    print(f"input text: {input}\ngenerated text: {generated_text}")

def generation_for_single_input_switch_embedd(model, encoding_embedd, encoder, decoder, input, do_sample, temperature, top_p, repetition_penalty):

    # get the device of the model
    embedding_device = model.get_input_embeddings().weight.device
    encoding_embedd = encoding_embedd.to(embedding_device)

    # encode the input
    tokenized_input = encoder(input, return_tensors='pt', return_attention_mask=True)

    # get the input_ids & attention_mask
    input_ids = tokenized_input['input_ids']
    input_ids = input_ids.to(embedding_device)
    attention_mask = tokenized_input['attention_mask']
    attention_mask = attention_mask.to(embedding_device)

    # encode the input id to input embedd by original word embedd
    inputs_embeds = encoding_embedd(input_ids)

    # generation & decoding
    generation = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=250, do_sample=do_sample, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    generated_text = decoder.decode(generation[0], skip_special_tokens=True)
    print(f"input text: {input}\ngenerated text: {generated_text}")  

def add_trans_promt_for_switch_embedd_unitest(src_lang, trg_lang, instruction):

    # promt format
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    inputs, targets = [prefix + sample + end_id + '\n' + assistant for sample in src_lang], [sample for sample in trg_lang]

    return inputs, targets

def check_numbers_of_trainable_parameters(model_path, token):

    # load the model
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                cache_dir='/export/data2/yleung/model_cache',
                                                device_map='cpu',
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                token=token)

    # check the # of trainable parameters
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()

    return trainable_params