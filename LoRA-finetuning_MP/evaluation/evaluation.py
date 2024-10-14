import os

# setting the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import json
import sacrebleu
import pandas as pd
from comet import download_model, load_from_checkpoint
import argparse

def comet_evaluation(file_dir, source_lang, target_lang):

  data_path = file_dir + '/' + source_lang + '-' + target_lang + '_comet'
  save_path = file_dir + '/COMET/' + source_lang + '-' + target_lang + '.comet'

  model_path = download_model("Unbabel/wmt22-comet-da")

  model = load_from_checkpoint(model_path)

  data_list = []

  with open(data_path, 'r') as file:
      for line in file:
        line = line.strip()
        if line:
          data = json.loads(line)
          data_list.append(data)

  model_output = model.predict(data_list, batch_size=32, gpus=1)

  with open(save_path, 'w') as f:
    f.write(f'COMET Score: {model_output.system_score}\n')

def corpus_bleu_evaluation(file_dir, source_lang, target_lang):

  data_path = file_dir + '/' + 'cleaned_' + source_lang + '-' + target_lang + '.csv'
  save_path = file_dir + '/BLEU/' + source_lang + '-' + target_lang + '.bleu'

  trans_corpus = []
  trg_corpus = []

  df = pd.read_csv(data_path)
  for i in range(len(df)):
    trans_corpus.append(df['predictions'].iloc[i])
    trg_corpus.append(df['labels'].iloc[i])

  # calculate the corpus bleu score
  bleu = sacrebleu.corpus_bleu(trans_corpus, [trg_corpus])

  # save the corpus bleu score
  with open(save_path, 'w') as f:
      f.write(f"BLEU Score: {bleu.score}\n")

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--directory', type=str, help='Directory for the results to be evaluate')
  parser.add_argument('-s', '--src_lang', type=str, default='eng_Latn', 
                    help='Source language (default: eng_Latn)')
  parser.add_argument('-t', '--trg_lang', type=str, default='deu_Latn', 
                    help='Target language (default: deu_Latn)')

  # get the arguments
  args = parser.parse_args()
  file_dir = args.directory
  source_lang = args.src_lang
  target_lang = args.trg_lang

  # evaluate the result with BLEU & COMET
  corpus_bleu_evaluation(file_dir, source_lang, target_lang)
  comet_evaluation(file_dir, source_lang, target_lang)