import torch
import numpy as np


if __name__ == '__main__':
    model_name = 'llava'
    alpha = 1
    suffix = '_sorry'
    jbv_idx_num = {}
    for layer_idx in [ 2, 3,4,5,6,7,8,9,11,12,13,14, 15,16,17,18,19,20,21,22]:
        mask_avg_jbv = np.load(f'masks/{model_name}/jbv_avg_{layer_idx}_alpha{alpha}{suffix}.npy')
        jbv_idx = np.where(mask_avg_jbv>=0.5)[0]
        jbv_idx_num[layer_idx] = len(jbv_idx)
        # seleted_jbv = mask_avg_jbv[seleted_idx]
        print(layer_idx, len(jbv_idx))
    sorted_keys = sorted(jbv_idx_num, key=jbv_idx_num.get)
    print(sorted_keys)
     