import pandas as pd
from datasets import load_dataset

dev = load_dataset('facebook/flores', 'yue_Hant')['dev']['sentence']
devtest = load_dataset('facebook/flores', 'yue_Hant')['devtest']['sentence']

cantonese_dev = {
    'dev': dev,
}
cantonese_devtest = {
    'devtest': devtest,
}
df = pd.DataFrame(cantonese_dev)
df.to_csv('flores_200_cantonese_dev.csv', index=False)
df = pd.DataFrame(cantonese_devtest)
df.to_csv('flores_200_cantonese_devtest.csv', index=False)