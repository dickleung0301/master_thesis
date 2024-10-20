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

def load_wmt19(dir):

    wmt19 = load_dataset('wmt/wmt19', dir, split='validation')

    return wmt19

def self_instruct(dataset, key, src_lang, trg_lang, instruction):

    # promt format
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    # return a list of input sequence with promt format
    return [prefix + sample[key][src_lang] + end_id + '\n' + assistant + sample[key][trg_lang] + end_id  for sample in dataset]

def get_max_length(dataset, tokenizer):

    # find the lengths of all seq
    len_seq = []
    for data in dataset:
        len_seq.append(len(tokenizer(data, truncation=False)['input_ids']))
    
    return max(len_seq)


def finetuning_preprocess(tokenizer, masking, dataset=None, key='translation', src_lang=None, trg_lang=None, trans_dir=None, split='train'): # 'translation' : Alma, 'direction': wmt22

    # load the config file & dict
    with open('config.json', 'r') as f:
        config = json.load(f)

    instruct = config['instruct']

    # instantiate the list of inputs & targets
    if dataset != None:
        inputs = self_instruct(dataset=dataset, key=key, src_lang=src_lang, trg_lang=trg_lang, instruction=instruct[trans_dir])
    else:
        # load all the translation directions & store them in a list
        list_alma_dataset = []
        if split == 'train':
            list_alma_dataset.append(load_alma(split=split, dir='cs-en'))
            list_alma_dataset.append(load_alma(split=split, dir='de-en'))
            list_alma_dataset.append(load_alma(split=split, dir='is-en'))
            list_alma_dataset.append(load_alma(split=split, dir='ru-en'))
            list_alma_dataset.append(load_alma(split=split, dir='zh-en'))
        else:
            list_alma_dataset.append(load_alma(split=split, dir='cs-en'))
            list_alma_dataset.append(load_alma(split=split, dir='de-en'))
            list_alma_dataset.append(load_alma(split=split, dir='ru-en'))
            list_alma_dataset.append(load_alma(split=split, dir='zh-en'))

        # apply llama promt format to all the translation direction and store in a list
        inputs = []
        for alma_dataset in list_alma_dataset:

            # get the key of the 1st & 2nd language
            first_lang = list(alma_dataset[0][key].keys())[0]
            second_lang = list(alma_dataset[0][key].keys())[1]

            # state the translation dir and reversed for getting the instruct from config
            trans_dir = first_lang + '-' + second_lang
            re_trans_dir = second_lang + '-' + first_lang

            # apply self-instruct format to the translation directions & append it to the inputs
            inputs.extend(self_instruct(dataset=alma_dataset, key=key, src_lang=first_lang, trg_lang=second_lang, instruction=instruct[trans_dir]))
            inputs.extend(self_instruct(dataset=alma_dataset, key=key, src_lang=second_lang, trg_lang=first_lang, instruction=instruct[re_trans_dir]))

    # set the pad token of llama & padding side
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # tokenize the data
    # be care of the truncation
    max_length = get_max_length(inputs, tokenizer)
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length + 1, return_attention_mask=True)

    # initialise a tensor in a shape of (dataset, 600) for masking & state the assistant turn for searching the end of it
    labels = torch.full((len(tokenized_inputs['input_ids']), max_length + 1), -100)
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"

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
    system = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
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