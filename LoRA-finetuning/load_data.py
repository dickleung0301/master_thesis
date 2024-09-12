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

class GenerationDataset(Dataset):

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def  __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
        }

def load_alma(split, dir):

    alma = load_dataset('haoranxu/ALMA-Human-Parallel', dir, split=split)

    return alma

def load_wmt22(dir):

    wmt22 = load_dataset('haoranxu/WMT22-Test', dir, split='test')

    return wmt22

def finetuning_preprocess(dataset, key, src_lang, trg_lang, tokenizer): # 'translation' : Alma, 'direction': wmt22

    # promt format & instruction
    instruction = "Translate Chinese to English: "
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    # instantiate the list of inputs & targets
    inputs = [prefix + sample[key][src_lang] + end_id + assistant for sample in dataset]
    targets = [sample[key][trg_lang] for sample in dataset]

    # set the pad token of llama & padding side
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # tokenize the data
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=1024, return_attention_mask=True)
    tokenized_targets = tokenizer(targets, padding='max_length', truncation=True, max_length=1024)

    # length of the prefix & the padding
    labels = torch.full((len(tokenized_targets['input_ids']), 1024), -100)
    for i, (input_ids, target_ids) in enumerate(zip(tokenized_inputs['input_ids'], tokenized_targets['input_ids'])):      

        # Find the position of the assistant token
        try:
            end_of_assistant_pos = len(input_ids) - 1 - input_ids[::-1].index(tokenizer.encode(assistant)[-1])
        except ValueError:
            print(f"Assistant token not found in sample {i}")
            continue
        
        # Calculate the number of non-pad tokens in the target
        target_length = sum(1 for id in target_ids if id != tokenizer.pad_token_id)
        
        # Assign the target ids to the labels, starting after the assistant token
        labels[i, end_of_assistant_pos + 1 : end_of_assistant_pos + 1 + target_length] = torch.tensor(
            [id for id in target_ids if id != tokenizer.pad_token_id]
        )

    # create the dataset
    dataset = TranslationDataset(
        input_ids = torch.tensor(tokenized_inputs['input_ids']),
        attention_mask = torch.tensor(tokenized_inputs['attention_mask']),
        labels = labels
    )

    return dataset

def generation_preprocess(dataset, key, src_lang, trg_lang, tokenizer): # 'translation' : Alma, 'direction': wmt22

    # promt format & instruction
    instruction = "Translate Chinese to English: "
    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>"
    system = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
    prefix = system + user + instruction

    # instantiate the list of inputs & targets
    inputs = [prefix + sample[key][src_lang] + end_id + assistant for sample in dataset]

    # set the pad token of llama & padding side
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    # tokenize the data
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=256, return_attention_mask=True)

    # create the dataset
    dataset = GenerationDataset(
        input_ids = torch.tensor(tokenized_inputs['input_ids']),
        attention_mask = torch.tensor(tokenized_inputs['attention_mask']),
    )

    return dataset