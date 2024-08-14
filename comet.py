import json
from comet import download_model, load_from_checkpoint

data_path = './zero_shot_result/eng_Latn2gle_Latn_comet'

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
print(model_output.system_score)