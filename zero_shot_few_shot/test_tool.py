import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM
from huggingface_hub import login

def print_trainable_parameters(model_choice):

    # load the access token
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    # load the model dict
    with open('config.json', 'r') as f:
        config = json.load(f)

    model = config["model"]

    # load the testing model
    model_name = model[model_choice]

    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True ,token=token)

    # check which parameters are trainable
    trainable_parameters = []
    non_trainable_parameters = []

    for name, param in pretrained_model.named_parameters():
        if param.requires_grad:
            trainable_parameters.append(name)
        else:
            non_trainable_parameters.append(name)

    # print out the trainable and non trainable param
    print("Trainable Parameters:")
    for name in trainable_parameters:
        print(name)

    print("Non Trainable Parameters:")
    for name in non_trainable_parameters:
        print(name)

def assert_embedding(model_choice):

    # load the access token
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    # to load the model dict
    with open('config.json', 'r') as f:
        config = json.load(f)

    model = config["model"]
    
    # load the model
    model_name = model[model_choice]
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True ,token=token)

    # do the assertion
    embed_tokens_weight = pretrained_model.model.embed_tokens.weight
    lm_head_weight = pretrained_model.lm_head.weight
    same = torch.equal(embed_tokens_weight, lm_head_weight)

    print("Are the word embedding & lm head the same?")
    print(same)