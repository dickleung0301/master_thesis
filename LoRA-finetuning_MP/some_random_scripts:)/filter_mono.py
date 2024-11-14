import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

mono_path = '/export/data2/yleung/smp_zht/train_sets/trainset_1/1.txt'
filter_mono_path = '/export/data2/yleung/smp_zht/train_sets/trainset_1/filter_32000_1.txt'
tokenizer_32000_path = '/export/data2/yleung/smp_zht/zht_32000'
tokenizer_64000_path = '/export/data2/yleung/smp_zht/zht_64000'

# load the access token from .env
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

# login huggingface_hub
login(token=token)

# define a function to draw pie chart
def pie_chart(array, name):

    bin_counts = {
        'first_bin': 0,
        'second_bin': 0,
        'third_bin': 0,
        'forth_bin': 0,
        'fifth_bin': 0
    }
    bin_size = 50

    # a dictionary to map the case to the bin
    bins = {
        0: 'first_bin',
        1: 'second_bin',
        2: 'third_bin',
        3: 'forth_bin',
    }

    for data in array:
        bin_number = data // bin_size
        bin_name = bins.get(bin_number, 'fifth_bin')
        bin_counts[bin_name] += 1
    
    labels = '0-49', '50-99', '100-149', '150-199', '>=200'
    sizes = [bin_counts['first_bin'], bin_counts['second_bin'], bin_counts['third_bin'], bin_counts['forth_bin'], bin_counts['fifth_bin']]
    fig, ax = plt.subplots()
    wedges, _ = ax.pie(sizes, startangle=140)

    # Add a legend with labels and percentages
    total = sum(sizes)
    legend_labels = [f"{label}: {size} ({size / total:.1%})" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Ranges", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.savefig(name, bbox_inches="tight")

        

# load the tokenizer 
custom_32000_tokenizer = AutoTokenizer.from_pretrained(tokenizer_32000_path, use_fast=False)
custom_64000_tokenizer = AutoTokenizer.from_pretrained(tokenizer_64000_path, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', cache_dir="/export/data2/yleung/model_cache", token=token)

# add the llama 3.1 Instruct special tokens
original_tokenizer_special_tokens = {
    "bos_token": tokenizer.bos_token,
    "eos_token": tokenizer.eos_token,
    "unk_token": tokenizer.unk_token,
    "pad_token": tokenizer.pad_token,
    "additional_special_tokens": list(tokenizer.added_tokens_decoder.values()),
}

custom_32000_tokenizer.add_special_tokens({
    "bos_token": original_tokenizer_special_tokens["bos_token"],
    "eos_token": original_tokenizer_special_tokens["eos_token"],
    "additional_special_tokens": original_tokenizer_special_tokens["additional_special_tokens"]
})

custom_64000_tokenizer.add_special_tokens({
    "bos_token": original_tokenizer_special_tokens["bos_token"],
    "eos_token": original_tokenizer_special_tokens["eos_token"],
    "additional_special_tokens": original_tokenizer_special_tokens["additional_special_tokens"]
})

num_tokens_llama = []
num_tokens_32000 = []
num_tokens_64000 = []

with open(mono_path, 'r') as infile: #, open(filter_mono_path, 'w') as outfile:
    for line in infile:
        line = line.strip()
        len_line_llama = len(tokenizer(line, truncation=False)['input_ids'])
        len_line_32000 = len(custom_32000_tokenizer(line, truncation=False)['input_ids'])
        len_line_64000 = len(custom_64000_tokenizer(line, truncation=False)['input_ids'])
        num_tokens_llama.append(len_line_llama)
        num_tokens_32000.append(len_line_32000)
        num_tokens_64000.append(len_line_64000)

        #if len_line_32000 <= 128:
        #    outfile.write(line + '\n')

#plt.hist(num_tokens_llama, bins=200, alpha=0.5, label='Llama', density=True)
#plt.hist(num_tokens_32000, bins=200, alpha=0.5, label='Vocab 32000', density=True)
#plt.hist(num_tokens_64000, bins=200)
#plt.xlabel('Number of Tokens')
#plt.ylabel('Frequency')
#plt.title('Distribution of # Tokens (Vocab 64000)')
#plt.legend()
#plt.savefig('tokens_dist_64000.png')
#plt.close()

pie_chart(num_tokens_llama, 'llama_pie_chart.png')
pie_chart(num_tokens_32000, '32000_pie_chart.png')
pie_chart(num_tokens_64000, '64000_pie_chart.png')