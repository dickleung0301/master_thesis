# master_thesis

## In-context learning results

### Zero-shot 

```bash
# an example script to run zero-shot baseline results for a language pair
python ./zsfs.py -i 'zero_shot' -s 'eng_Latn' -t 'deu_Latn' -m '6'
# results in ./zero_shot_result
```
### Few-shot 
```bash
# an example script to run few-shot baseline results for a language pair
python ./zsfs.py -i 'few_shot' -n 3 -s 'eng_Latn' -t 'deu_Latn' -m '6'
# results in ./few_shot_in_context_result
```
### Fine-tuning
```bash
# an example script to fine-tune llama3 for mt
python ./main.py -m 1 -s 'zh' -t 'en' --train --no-inference -lr 1e-5 -n 3 -sd './'
# the model will then be saved in './'
```
### Generation for the fine-tuned model
```bash
# an example script to fine-tune llama3 for mt
python ./main.py -m 1 -s 'zh' -t 'en' --no-train --inference -lr 1e-5 -n 3 -sd './'
# the prediction will then be saved in './prediction.csv'
```
