import json
import sacrebleu
from comet import download_model, load_from_checkpoint
import argparse

def comet_evaluation(file_dir, source_lang, target_lang):

  data_path = file_dir + '/' + source_lang + '2' + target_lang + '_comet'
  save_path = file_dir + '/COMET/' + source_lang + '2' + target_lang + '.comet'

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

def corpus_bleu_evaluation(file_dir, source_lang, target_lang, bleu_tokenize):

  trans_path = file_dir + '/' + source_lang +'2' + target_lang + '_trans.txt'
  trg_path = file_dir + '/' + source_lang +'2' + target_lang + '_trg.txt'
  save_path = file_dir + '/BLEU/' + source_lang + '2' + target_lang + '.bleu'

  # read the translation and target file
  with open(trans_path, 'r') as f:
      trans_corpus = [f.read()]

  with open(trg_path, 'r') as f:
      trg_corpus = [f.read()]

  # calculate the corpus bleu score
  bleu = sacrebleu.corpus_bleu(trans_corpus, [trg_corpus], tokenize=bleu_tokenize)

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
  parser.add_argument('-bt', '--bleu_tokenize', type=str, default='none',
                    help='Tokenize method for BLEU (default: none)')

  # get the arguments
  args = parser.parse_args()
  file_dir = args.directory
  source_lang = args.src_lang
  target_lang = args.trg_lang
  bleu_tokenize = args.bleu_tokenize

  # evaluate the result with BLEU & COMET
  corpus_bleu_evaluation(file_dir, source_lang, target_lang, bleu_tokenize)
  comet_evaluation(file_dir, source_lang, target_lang)