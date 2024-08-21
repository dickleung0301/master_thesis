import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from exception import *

# load the source & target language
def load_flores200(split, source_lang, target_lang, prefix_L1, prefix_L2):

    try:
        flores200_src = load_dataset('facebook/flores', source_lang)[split]
        temp = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
        temp2 = '\n' + '<|start_header_id|>assistant<|end_header_id|>'
        data = {}
        data['id'] = flores200_src['id']
        data['src'] = flores200_src['sentence']
        data[source_lang] = [temp + prefix_L1 + x + ' = ' + prefix_L2 + temp2 for x in flores200_src['sentence']]
        data[target_lang] = load_dataset('facebook/flores', target_lang)[split]['sentence']

        return data

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in loading flores-200: {e}")

# load the source & target language for few-shot learning
def load_flores200_few_shot(split, source_lang, target_lang, prefix_L1, prefix_L2, num_samples):

    try:
        # load the source lang & target lang
        flores200_src = load_dataset('facebook/flores', source_lang)[split]
        flores200_trg = load_dataset('facebook/flores', target_lang)[split]

        # sample the examples for few-shot learning
        total_num_sample = len(flores200_src)
        example_idx = np.random.randint(low=0, high=total_num_sample, size=num_samples)

        # constructing the dictionary for few-shot learning
        data = {
            'id': [],
            'src': [],
            source_lang: [],
            target_lang: [],
        }

        # the prompt format of llama 3
        temp = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
        temp2 = '\n' + '<|start_header_id|>assistant<|end_header_id|>'

        for idx in example_idx:
            data['id'].append(flores200_src['id'][idx])
            data['src'].append(flores200_src['sentence'][idx])
            data[source_lang].append(temp + prefix_L1 + flores200_src['sentence'][idx] + ' = ' + prefix_L2 + temp2)
            data[target_lang].append(flores200_trg['sentence'][idx])

        return data 

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in loading flores-200 for few-shot: {e}")

def load_flores200_few_shot_in_context(split, example_split, source_lang, target_lang, prefix_L1, prefix_L2, num_example):

    try:
        # load the source lang & target lang
        flores200_src = load_dataset('facebook/flores', source_lang)[split]
        flores200_trg = load_dataset('facebook/flores', target_lang)[split]

        # load the sorce lang & target lang for example split
        example_src = load_dataset('facebook/flores', source_lang)[example_split]
        example_trg = load_dataset('facebook/flores', target_lang)[example_split]

        # the index of the examples
        example_idx = np.arange(len(example_src))

        # constructing the dictionary for few-shot learning
        data = {
            source_lang: [],
            target_lang: [],
        }

        data['id'] = flores200_src['id']
        data['src'] = flores200_src['sentence']
        data[target_lang] = flores200_trg['sentence']

        for i in range(len(flores200_src)):
            chosen_examples = np.random.choice(example_idx, size=num_example, replace=False)
            user = "<|start_header_id|>user<|end_header_id|>\n"
            assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
            end_id = "<|eot_id|>"
            temp = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
            for j in chosen_examples:
                temp += user + prefix_L1 + example_src['sentence'][j] + '\n' + end_id + assistant + prefix_L2 + example_trg['sentence'][j] + '\n' + end_id
            temp += user + prefix_L1 + flores200_src['sentence'][i] + prefix_L2 + '\n' + end_id + assistant
            
            data[source_lang].append(temp)
        
        return data

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in loading flores-200 for few-shot in context: {e}")

# load data for sanity check
def load_flores200_sanity_check(split, source_lang, target_lang, prefix_L1, prefix_L2, index):

    try:
        flores200_src = load_dataset('facebook/flores', source_lang)[split]
        flores200_trg = load_dataset('facebook/flores', target_lang)[split]
        temp = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
        temp2 = '\n' + '<|start_header_id|>assistant<|end_header_id|>'
        data = {}
        data['id'] = [flores200_src[index]['id']]
        data['src'] = [flores200_src[index]['sentence']]
        data[source_lang] = [temp + prefix_L1 + flores200_src[index]['sentence'] + ' = ' + prefix_L2 + temp2]
        data[target_lang] = [flores200_trg[index]['sentence']]

        return data        
    except Exception as e:
        raise DataLoadingError(f"There is something wrong in loading flores-200 for sanity check: {e}")

def tokenize_data(data, source_lang, target_language, tokenizer, truncation, MAX_LEN):

    try:
        data['input_ids'] = [tokenizer.encode(x, truncation=truncation, padding="max_length", max_length=MAX_LEN)
                            for x in data[source_lang]]
        data['target_ids'] = [tokenizer.encode(x, truncation=truncation, padding="max_length", max_length=MAX_LEN)
                            for x in data[target_language]]
        data['src_ids'] = [tokenizer.encode(x, truncation=truncation, padding="max_length", max_length=MAX_LEN)
                            for x in data['src']]

        return data

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in tokenization: {e}")

def create_dataloader(input_ids, attention_masks, target_ids, src_ids, batch_size, shuffle=True):

    try:
        dataset = TensorDataset(input_ids, attention_masks, target_ids, src_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in creating the dataloader: {e}")