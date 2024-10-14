import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# define the custom dataset
class TranslationDataset(Dataset):

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def  __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'labels': self.labels[index],
        }

def load_alma(split, dir):

    alma = load_dataset('haoranxu/ALMA-Human-Parallel', dir, split=split)

    return alma

def load_wmt22(dir):

    wmt22 = load_dataset('haoranxu/WMT22-Test', dir, split='test')

    return wmt22

def finetuning_preprocess(dataset, key, src_lang, trg_lang, trans_dir, tokenizer, masking): # 'translation' : Alma, 'direction': wmt22

    # load the config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    instruct = config['instruct']

    # promt format & instruction
    instruction = instruct[trans_dir]
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    # instantiate the list of inputs & targets
    inputs = [prefix + sample[key][src_lang] + end_id + '\n' + assistant + sample[key][trg_lang] + end_id  for sample in dataset]

    # set the pad token of llama & padding side
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # tokenize the data
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=600, return_attention_mask=True)

    # initialise a tensor in a shape of (dataset, 600) for masking
    labels = torch.full((len(tokenized_inputs['input_ids']), 600), -100)
    for i, input_ids in enumerate(tokenized_inputs['input_ids']):      

        # Find the position of the assistant token
        try:
            end_of_assistant_pos = len(input_ids) - 1 - input_ids[::-1].index(tokenizer.encode(assistant)[-1])
        except ValueError:
            print(f"Assistant token not found in sample {i}")
            continue
        
        # Calculate the number of non-pad tokens in the input
        # input_length = sum(1 for id in input_ids if id != tokenizer.pad_token_id)
        

        if masking:
            # Assign the target ids to the labels, starting after the assistant token
            labels[i, end_of_assistant_pos :] = torch.tensor(
                input_ids[end_of_assistant_pos :]
            )
        else:
            # Assign the target ids to the labels, without making
            labels[i, 0 :] = torch.tensor(
                input_ids[0 :]
            )

    # create the dataset
    dataset = TranslationDataset(
        input_ids = torch.tensor(tokenized_inputs['input_ids']),
        attention_mask = torch.tensor(tokenized_inputs['attention_mask']),
        labels = labels
    )

    return dataset

def generation_preprocess(dataset, key, src_lang, trg_lang, trans_dir, tokenizer, right_padding): # 'translation' : Alma, 'direction': wmt22

    # load the config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    instruct = config['instruct']

    # promt format & instruction
    instruction = instruct[trans_dir]
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    # instantiate the list of inputs & targets
    inputs = [prefix + sample[key][src_lang] + end_id + '\n' + assistant for sample in dataset]
    labels = [sample[key][trg_lang] for sample in dataset]

    # set the pad token of llama & padding side
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if not right_padding:
        tokenizer.padding_side = 'left'
    else:
        tokenizer.padding_side = 'right'

    # tokenize the data
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=128, return_attention_mask=True)
    tokenized_labels = tokenizer(labels, padding='max_length', truncation=True, max_length=128)

    # create the dataset
    dataset = TranslationDataset(
        input_ids = torch.tensor(tokenized_inputs['input_ids']),
        attention_mask = torch.tensor(tokenized_inputs['attention_mask']),
        labels = torch.tensor(tokenized_labels['input_ids'])
    )

    return dataset