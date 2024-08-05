# a function to post-process the generation from llama
def strip_llama_output(generation):
    punctuation_to_stop = '.'
    promt_format = 'assistant'
    striped_generation = generation.split(promt_format)[-1]
    striped_generation = striped_generation.replace('"', '') # edge case for generating "
    output = striped_generation.split(punctuation_to_stop)[0]
    output += punctuation_to_stop
    output = output.strip()

    return output

# a function to post-process the generation of in-context learning for llama
def strip_in_context(generation, prefix_L2):
    phase_to_strip = prefix_L2
    striped_generation = generation.split(phase_to_strip)[-1]

    return striped_generation

# a function to save the corpus
def save_corpus(corpus, save_directory, source_lang, target_lang, translation=True):
    if translation:
        temp = '_trans'
    else:
        temp ='_trg'
    with open(save_directory + '/' + source_lang + '2' + target_lang + temp + '.txt', 'w') as file:
        file.write(corpus)