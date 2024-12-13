import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["HF_DATASETS_CACHE"] = "/export/data2/yleung/dataset"

import json
import torch
import copy
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

def split_yue_paragraph_into_sentence(en_yue):

    # to instantiate two empty lists for sentence level corpus
    length_of_dataset = len(en_yue['en'])
    en = []
    yue = []

    # to split the parallel corpus into sentence level
    for i in range(length_of_dataset):
        sentence_level_en = en_yue['en'][i].split('.')
        sentence_level_yue = en_yue['yue'][i].split('。')

        sentence_level_en = [item.strip() + '.' for item in sentence_level_en]
        sentence_level_yue = [item.strip() + '。' for item in sentence_level_yue]

        # to kick out the unmatched entry
        if len(sentence_level_en) == len(sentence_level_yue):
            en.extend(sentence_level_en)
            yue.extend(sentence_level_yue)

    return en, yue

def load_yue_trans():

    yue_tran = load_dataset('BillBao/Yue-Benchmark', 'Yue-TRANS', split='test')
    en_yue = yue_tran[:200]

    en, yue = split_yue_paragraph_into_sentence(en_yue)

    return {
            'en': en,
            'yue': yue,
    }

def load_flores(source_lang, trg_lang, split):

    # load the language iso code mapping for flores
    with open('config.json', 'r') as f:
        config = json.load(f)

    lang_iso_code_mapping = config['flores']

    flores200_src = load_dataset('facebook/flores', lang_iso_code_mapping[source_lang], trust_remote_code=True)[split]['sentence']
    flores200_trg = load_dataset('facebook/flores', lang_iso_code_mapping[trg_lang], trust_remote_code=True)[split]['sentence']

    return {
            source_lang: flores200_src,
            trg_lang: flores200_trg,
    }

def chat_temp_llama_Instruct_finetune(dataset, key, src_lang, trg_lang, instruction):

    # promt format
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    # return a list of input sequence with promt format
    return [prefix + sample[key][src_lang] + end_id + '\n' + assistant + sample[key][trg_lang] + end_id  for sample in dataset]

def chat_temp_llama_Instruct_generation(dataset, key, src_lang, trg_lang, instruction):

    # promt format
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    if key != None:
        inputs, targets = [prefix + sample[key][src_lang] + end_id + '\n' + assistant for sample in dataset], [sample[key][trg_lang] for sample in dataset]
    else:
        inputs, targets = [prefix + dataset[src_lang][i] + end_id + '\n' + assistant for i in range(len(dataset[src_lang]))], [dataset[trg_lang][i] for i in range(len(dataset[trg_lang]))]

    # return a list of input sequence with promt format
    return inputs, targets

def chat_temp_tower_Instruct_finetune(dataset, key, src_lang, trg_lang, instruction):

    # promt format
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    end_id = "<|im_end|>"
    prefix = user + '\n' + instruction

    # return a list of input sequence with promt format
    return [prefix + sample[key][src_lang] + end_id + '\n' + assistant + '\n' + sample[key][trg_lang] + end_id for sample in dataset]

def chat_temp_tower_Instruct_generation(dataset, key, src_lang, trg_lang, instruction):

    # promt format
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    end_id = "<|im_end|>"
    prefix = user + '\n' + instruction

    # return a list of input sequence with promt format
    return [prefix + sample[key][src_lang] + end_id + '\n' + assistant for sample in dataset], [sample[key][trg_lang] for sample in dataset]

def get_max_length(dataset, tokenizer):

    # find the lengths of all seq
    len_seq = []
    for data in dataset:
        len_seq.append(len(tokenizer(data, truncation=False)['input_ids']))
    
    return max(len_seq)


def finetuning_preprocess(model_choice, tokenizer, masking, dataset=None, key='translation', src_lang=None, trg_lang=None, trans_dir=None, split='train'): # 'translation' : Alma, 'direction': wmt22

    # load the config file & dict
    with open('config.json', 'r') as f:
        config = json.load(f)

    instruct = config['instruct']

    # instantiate the list of inputs & targets
    if dataset != None:
        if model_choice == 1 or model_choice == 2:
            inputs = chat_temp_llama_Instruct_finetune(dataset=dataset, key=key, src_lang=src_lang, trg_lang=trg_lang, instruction=instruct[trans_dir])
        elif model_choice == 3:
            inputs = chat_temp_tower_Instruct_finetune(dataset=dataset, key=key, src_lang=src_lang, trg_lang=trg_lang, instruction=instruct[trans_dir])
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
            if model_choice == 1 or model_choice == 2:
                inputs.extend(chat_temp_llama_Instruct_finetune(dataset=alma_dataset, key=key, src_lang=first_lang, trg_lang=second_lang, instruction=instruct[trans_dir]))
                inputs.extend(chat_temp_llama_Instruct_finetune(dataset=alma_dataset, key=key, src_lang=second_lang, trg_lang=first_lang, instruction=instruct[re_trans_dir]))
            elif model_choice == 3:
                inputs.extend(chat_temp_tower_Instruct_finetune(dataset=alma_dataset, key=key, src_lang=first_lang, trg_lang=second_lang, instruction=instruct[trans_dir]))
                inputs.extend(chat_temp_tower_Instruct_finetune(dataset=alma_dataset, key=key, src_lang=second_lang, trg_lang=first_lang, instruction=instruct[re_trans_dir]))

    # set the pad token of llama & padding side
    if model_choice == 1 or model_choice == 2:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # tokenize the data
    # be care of the truncation
    max_length = get_max_length(inputs, tokenizer)
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length + 1, return_attention_mask=True)

    # initialise a tensor in a shape of (dataset, max_length) for masking & state the assistant turn for searching the end of it
    labels = torch.full((len(tokenized_inputs['input_ids']), max_length + 1), -100)

    if model_choice == 1 or model_choice == 2:
        assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    elif model_choice == 3:
        assistant = "<|im start|>assistant\n"

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

def generation_preprocess(model_choice, dataset, src_lang, trg_lang, trans_dir, tokenizer, right_padding, key=None): # 'translation' : Alma, 'direction': wmt22

    # load the config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    instruct = config['instruct']

    # instantiate the list of inputs & targets
    if model_choice == 1 or model_choice == 2:
        inputs, labels = chat_temp_llama_Instruct_generation(dataset=dataset, key=key, src_lang=src_lang, trg_lang=trg_lang, instruction=instruct[trans_dir])
    elif model_choice == 3:
        inputs, labels = chat_temp_tower_Instruct_generation(dataset=dataset, key=key, src_lang=src_lang, trg_lang=trg_lang, instruction=instruct[trans_dir])

    # set the pad token of llama & padding side
    if model_choice == 1 or model_choice == 2:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not right_padding:
        tokenizer.padding_side = 'left'
    else:
        tokenizer.padding_side = 'right'

    # check the max seq len of inputs & labels
    max_length_inputs = get_max_length(inputs, tokenizer)
    max_length_labels = get_max_length(labels, tokenizer)

    if max_length_inputs >= max_length_labels:
        max_length = max_length_inputs
    else:
        max_length = max_length_labels

    # tokenize the data
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length, return_attention_mask=True)
    tokenized_labels = tokenizer(labels, padding='max_length', truncation=True, max_length=max_length)

    # create the dataset
    dataset = TranslationDataset(
        input_ids = torch.tensor(tokenized_inputs['input_ids']),
        attention_mask = torch.tensor(tokenized_inputs['attention_mask']),
        labels = torch.tensor(tokenized_labels['input_ids'])
    )

    return dataset