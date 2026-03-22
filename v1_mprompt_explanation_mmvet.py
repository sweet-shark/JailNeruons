import torch
import os
import pandas as pd
import json


from MLLM_models import *
from v1_expalanation_utils import *
from tqdm import tqdm


model = LLaVA()
# print(model.vl_gpt)
# raise
model_name = 'llava'
mmvet_path = '/path/to/your/mm-vet'
model.eval()

test_exp = init_exp(model_name, model)

image_folder = os.path.join(mmvet_path, "images")
meta_data = os.path.join(mmvet_path, "mm-vet.json")
with open(meta_data, 'r') as f:
    data = json.load(f)

# try:
#     forward_info = torch.load(f'/storage/gyy/mllm/LLM_EXPLANATION/new_forward_info/forward_info_mmvet_{model_name}_temp.pt')
# except Exception as e:
#     print('e is ',e)
#     forward_info = torch.load(f'/storage/gyy/mllm/LLM_EXPLANATION/new_forward_info/forward_info_mmvet_{model_name}.pt')
# test_exp.forward_info = forward_info

for i in tqdm(range(len(data))):
    if i < len(test_exp.forward_info) or i == 206 or i == 110:
        continue
    idx = f"v1_{i}"
    imagename = data[idx]['imagename']
    img_path = os.path.join(image_folder, imagename)
    prompt = data[idx]['question']
    with torch.no_grad():
        results = model.inference(img_path, prompt, generate=False)
    test_exp.get_forward_info(results, 2, pos = -1)
    torch.save(test_exp.forward_info, f'hiddenstates/forward_info_mmvet_{model_name}.pt')
    torch.save(test_exp.forward_info, f'hiddenstates/forward_info_mmvet_{model_name}_temp.pt') #During the torch.save process, you may encounter a memory full issue that leads to termination. Therefore, saving two copies can prevent file corruption in case the save fails.

    