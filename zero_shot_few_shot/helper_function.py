import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

# a function to post-process the generation from llama
def strip_zero_shot(generation):
    promt_format = 'assistant\n'
    striped_generation = generation.split(promt_format)[-1]
    striped_generation = striped_generation.strip()

    return striped_generation

# a function to post-process the generation of in-context learning for llama
def strip_in_context(generation, prefix_L2):
    phase_to_strip = prefix_L2
    striped_generation = generation.split(phase_to_strip)[-1]
    striped_generation = striped_generation.strip()

    return striped_generation

# a function to save the corpus
def save_corpus(corpus, save_directory, source_lang, target_lang, translation=True, original=False):
    if translation and not original:
        temp = '_trans'
    elif translation and original:
        temp = '_original_output'
    elif not translation:
        temp ='_trg'
    with open(save_directory + '/' + source_lang + '2' + target_lang + temp + '.txt', 'w') as file:
        file.write(corpus)

# a function to apply the promt format
def apply_chat_template(prefix_L1, prefix_L2, src_lang):

    system = "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant for translations\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
    assistant = '<|start_header_id|>assistant<|end_header_id|>\n'
    end_id = "<|eot_id|>\n"

    return system + prefix_L1 + src_lang + '\n' + prefix_L2 + end_id + assistant

def apply_chat_in_context(text, prefix_L1, prefix_L2, src_lang, trg_lang = None):

    user = "<|start_header_id|>user<|end_header_id|>\n"
    assistant = "<|start_header_id|>assistant<|end_header_id|>\n"
    end_id = "<|eot_id|>\n"

    if trg_lang is not None:
        text = text + user + prefix_L1 + src_lang + '\n' + end_id + assistant + prefix_L2 + trg_lang + '\n' + end_id
    else:
        text = text + user + prefix_L1 + src_lang + '\n' + end_id + assistant
