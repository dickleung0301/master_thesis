import pandas as pd
import re
import sacrebleu

pred = 'cleaned_v3_fine-tuned_de_en_grad_accumpredictions.csv'

def strip_after_colon_slash(text):
    matches = re.findall(r'assistant\s*(.*)', text)
    if matches:
        return matches[-1]  # Return the last match from the list
    return None  # Return None if no match is found

df = pd.read_csv(pred)

df['cleaned_predictions'] = df['predictions'].apply(strip_after_colon_slash)

references = [df['labels'].tolist()]
hypotheses = df['cleaned_predictions'].tolist()

bleu = sacrebleu.corpus_bleu(hypotheses, references)

print(f"SacreBLEU Score: {bleu.score:.4f}")

df.to_csv('strip_' + pred, index=False)