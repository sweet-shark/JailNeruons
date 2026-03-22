# This 

import torch
import numpy as np
import matplotlib.pyplot as plt

model_name = 'llava'
map_idx = torch.load(f'/path/to/your/JailBreakV_28k/{model_name}/attack_success.pt')
attack = 'jbv'
suffix = '_sorry'
# mask_nps = []
for alpha in [1]:
    # for layer_idx in [5, 8, 9, 10, 11, 16]:
    for layer_idx in [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20, 21,22]:
    # for layer_idx in [13, 17, 21, 25]:
        # layer_idx = 5
        mask_nps = []
        num_nan = 0
        for j in range(len(map_idx)):
            idx = map_idx[j]
            mask = torch.load(f'masks/{model_name}/{attack}_{idx}_{layer_idx}_alpha{alpha}{suffix}.pt').to(torch.float32)
            if torch.isnan(mask).any():
                num_nan += 1
                continue
            mask = torch.sigmoid(mask.detach().clone())
            mask_np = mask.cpu().numpy()[0,0]
            mask_nps.append(mask_np)

        mask_nps = np.stack(mask_nps,axis=0)    
        mask_avg = np.mean(mask_nps, axis=0)
        # plt.figure()
        # plt.hist(mask_avg, bins=20)
        # plt.savefig(f'vis/MaskAvg_{model_name}_{layer_idx}_alpha{alpha}_{j}.png')
        # print(len(np.where(mask_avg>=0.05)[0]))
        print(num_nan, layer_idx, mask_avg.max())
        # plt.figure()
        # plt.hist(mask_avg, bins=20)
        # plt.savefig(f'vis/mask_{layer_idx}_alpha{alpha}.png')
        # print(mask_avg.shape)
        np.save(f'masks/{model_name}/{attack}_avg_{layer_idx}_alpha{alpha}{suffix}', mask_avg)