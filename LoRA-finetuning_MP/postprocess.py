import pandas as pd
import re

pred = 'cleaned_v2_fine-tuned_de_en_grad_accumpredictions.csv'
save_path = 'cleaned_v3_fine-tuned_de_en_grad_accumpredictions.csv'

# load the prediction
df = pd.read_csv(pred)

# a functions to strip the repetition
def remove_repeated_phrases(text):
    # Pattern to capture repeated words or phrases with punctuation
    # This pattern tries to capture sequences of words, numbers, or punctuation that repeat.
    pattern = r'(\b[\w.,:;!?]+\b)( \1)+'
    
    # Replace any repeated words or phrases with the first instance
    return re.sub(pattern, r'\1', text)

# Function to remove longer repeated sequences, like multiple words or numbers
def remove_long_repetitions(text):
    # This pattern will capture sequences that repeat (e.g., "249. 249." or "value. value. value.")
    pattern = r'(\b(?:[\w.,:;!?]+\s*){1,5})(?:\1)+'
    return re.sub(pattern, r'\1', text)

# First, apply the basic repetition removal function
df['predictions'] = df['predictions'].apply(lambda x: remove_repeated_phrases(str(x)))

# Then, apply the function that targets longer sequences
df['predictions'] = df['predictions'].apply(lambda x: remove_long_repetitions(str(x)))

df.to_csv(save_path, index=False)