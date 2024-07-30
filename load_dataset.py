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
            source_lang: [],
            target_lang: [],
        }

        # the prompt format of llama 3
        temp = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
        temp2 = '\n' + '<|start_header_id|>assistant<|end_header_id|>'

        for idx in example_idx:
            data['id'].append(flores200_src['id'][idx])
            data[source_lang].append(temp + prefix_L1 + flores200_src['sentence'][idx] + ' = ' + prefix_L2 + temp2)
            data[target_lang].append(flores200_trg['sentence'][idx])

        return data 

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in loading flores-200 for few-shot: {e}")

def load_flores200_few_shot_in_context(split, source_lang, target_lang, prefix_L1, prefix_L2, num_example, num_inference):

    try:
        # load the source lang & target lang
        flores200_src = load_dataset('facebook/flores', source_lang)[split]
        flores200_trg = load_dataset('facebook/flores', target_lang)[split]

        # sample the idx for the sentence we want to translate
        total_num_sample = len(flores200_src)
        translate_idx = np.random.randint(low=0, high=total_num_sample, size=num_inference)

        # to get the idx of the dataset excluding those we took it for translation
        idx = np.arange(0, total_num_sample)
        example_idx = np.setdiff1d(idx, translate_idx)

        # constructing the dictionary for few-shot learning
        data = {
            'id': [],
            source_lang: [],
            target_lang: [],
        }

        for i in translate_idx:
            chosen_examples = np.random.choice(example_idx, size=num_example, replace=False)
            temp = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
            temp2 = []
            for j in chosen_examples:
                temp += prefix_L1 + flores200_src['sentence'][j] + ' = ' + prefix_L2 + flores200_trg['sentence'][j] + '\n'
                temp2.append(j)
            temp += prefix_L1 + flores200_src['sentence'][i] + ' = ' + prefix_L2 + '\n' +' <|eot_id|>'
            temp += '\n' + '<|start_header_id|>assistant<|end_header_id|>'
            temp2.append(i)
            
            data['id'].append(temp2)
            data[source_lang].append(temp)
            data[target_lang].append(flores200_trg['sentence'][i])
        
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

        return data

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in tokenization: {e}")

def create_dataloader(input_ids, attention_masks, target_ids, batch_size, shuffle=True):

    try:
        dataset = TensorDataset(input_ids, attention_masks, target_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in creating the dataloader: {e}")