# a function to post-process the generation from llama
def strip_zero_shot(generation):
    punctuation_to_stop = '.'
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