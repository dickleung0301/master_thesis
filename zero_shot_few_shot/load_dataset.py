import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import numpy as np
from helper_function import apply_chat_template, apply_chat_in_context
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from exception import *

# load the source & target language
def load_flores200(split, source_lang, target_lang, prefix_L1, prefix_L2):

    try:
        flores200_src = load_dataset('facebook/flores', source_lang)[split]
        data = {}
        data['id'] = flores200_src['id']
        data['src'] = flores200_src['sentence']
        data[source_lang] = [apply_chat_template(prefix_L1, prefix_L2, x) for x in flores200_src['sentence']]
        data[target_lang] = load_dataset('facebook/flores', target_lang)[split]['sentence']

        return data

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in loading flores-200: {e}")

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
            input = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n"
            for j in chosen_examples:
                apply_chat_in_context(input, prefix_L1, prefix_L2, example_src['sentence'][j], trg_lang = example_trg['sentence'][j])
            apply_chat_in_context(input, prefix_L1, prefix_L2, flores200_src['sentence'][i])
            
            data[source_lang].append(input)
        
        return data

    except Exception as e:
        raise DataLoadingError(f"There is something wrong in loading flores-200 for few-shot in context: {e}")

# load data for sanity check
def load_flores200_sanity_check(split, source_lang, target_lang, prefix_L1, prefix_L2, index):

    try:
        flores200_src = load_dataset('facebook/flores', source_lang)[split]
        flores200_trg = load_dataset('facebook/flores', target_lang)[split]
        data = {}
        data['id'] = [flores200_src[index]['id']]
        data['src'] = [flores200_src[index]['sentence']]
        data[source_lang] = [apply_chat_template(prefix_L1, prefix_L2, flores200_src[index]['sentence'])]
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