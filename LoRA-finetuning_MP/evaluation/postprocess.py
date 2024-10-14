import json
import pandas as pd
import argparse

# a function to strip the inputs & generations of LLMs
def strip_output(path, src_lang, trg_lang, save_dir):
    
    # get the instruction from config
    dir = src_lang + '-' + trg_lang
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    instruct = config['instruct'] 

    df = pd.read_csv(path)

    for i in range(len(df)):
        # strip the inputs
        df['inputs'].iloc[i] = df['inputs'].iloc[i].split('assistant\n')[0]
        df['inputs'].iloc[i] = df['inputs'].iloc[i].split(instruct[dir])[-1]
        df['inputs'].iloc[i] = df['inputs'].iloc[i].strip()
        # strip the generations
        df['predictions'].iloc[i] = df['predictions'].iloc[i].split('assistant\n')[-1]
        df['predictions'].iloc[i] = df['predictions'].iloc[i].strip()

    save_dir = save_dir + '/' + 'cleaned_' + dir + '.csv'
    df.to_csv(save_dir, index=False)

    return save_dir

# a function to format the input for comet 
def format_comet(path, src_lang, trg_lang, save_dir):

    df = pd.read_csv(path)
    comet_eval = []

    # assign every entry of df to a format for comet evaluation
    for i in range(len(df)):
        data = {
            "src": df['inputs'].iloc[i],
            "mt": df['predictions'].iloc[i],
            "ref": df['labels'].iloc[i]
        }
        comet_eval.append(data)

    # save the data struct for comet
    output_file = open(save_dir + '/' + src_lang + '-' + trg_lang + '_comet', 'w', encoding='utf-8')
    for data in comet_eval:
        json.dump(data, output_file)
        output_file.write("\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_lang', type=str, help='source language of the translation direction')
    parser.add_argument('-t', '--trg_lang', type=str, help='target language of the translation direction')
    parser.add_argument('-p', '--path', type=str, help='path of the raw data')
    parser.add_argument('-sd', '--save_dir', type=str, help='saving directory of the processed data')

    args = parser.parse_args()
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    path = args.path
    save_dir = args.save_dir

    processed_path = strip_output(path=path, src_lang=src_lang, trg_lang=trg_lang, save_dir=save_dir)
    format_comet(path=processed_path, src_lang=src_lang, trg_lang=trg_lang, save_dir=save_dir)