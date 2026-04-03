"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""

import torch
import numpy as np
from torch.cuda.amp import autocast as autocast
import random
from load_data import get_data
import matplotlib.pyplot as plt

from v1_expalanation_utils import *
from sklearn.svm import OneClassSVM


def process_data(forward_info, topk_indices, layer_idx,):
    new_forward_info = {}
    for k, v in forward_info.items():
        new_forward_info[k] = {"hidden_states": v["hidden_states"][layer_idx], "top-value_pair": v["top-value_pair"][layer_idx], "label":v["label"]}
    features = []
    labels = []
    for key, value in new_forward_info.items():
        for hidden_state in value["hidden_states"]:
            if topk_indices is not None:
                seleted_featrues = hidden_state.flatten()[topk_indices]
            else:
                seleted_featrues = hidden_state.flatten()
            features.append(seleted_featrues)
            labels.append(value["label"])
    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def train_with_reference(hiddenstates, layer_nums, data_type, top_k = 10, target_layers=None, alpha=1, thre=None, surffix=''):
    if target_layers is None:
        target_layers = range(0, layer_nums-1)
        
    features_all_data = []
    labels_all_data = []
    
    for hiddenstate in hiddenstates:
        features_all_layer = []
        for j in target_layers:
            top_indices_all_data = []
            for data_t in data_type:
                mask_avg = np.load(f'/path/to/mask_avg')
                if thre is None:
                    if top_k:
                        topk_indices = np.argsort(mask_avg)[-top_k:]
                    else:
                        raise
                else:
                    topk_indices = np.where(mask_avg>=thre)[0]
                top_indices_all_data.append(topk_indices)
            top_indices_all_data = np.unique(np.concatenate(top_indices_all_data, axis=0))
            features_ae, labels_ae = process_data(hiddenstate, top_indices_all_data, j,)
            features_all_layer.append(features_ae)
        labels_all_data.append(labels_ae)
        features_all_layer = np.concatenate(features_all_layer, axis=1)
        features_all_data.append(features_all_layer)
    features_all_data = np.concatenate(features_all_data, axis=0)
    labels_all_data = np.concatenate(labels_all_data, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        features_all_data, labels_all_data, test_size=0.5, random_state=42
    )
    
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train, y_train)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        alpha=0.01,
        solver="adam",
        verbose=0,
        random_state=42,
        learning_rate_init=0.01,
    )
    mlp_model.fit(X_train_scaled, y_train)
    
    svm_test = svm_model.predict(X_test)
    # print(svm_test, y_test)
    X_test_scaled = scaler.transform(X_test)
    mlp_test = mlp_model.predict(X_test_scaled)
    # print(sum(svm_test == y_test)/len(y_test))
    # print(sum(mlp_test == y_test)/len(y_test))
    return svm_model, sum(svm_test == y_test)/len(y_test), mlp_model, scaler, sum(mlp_test == y_test)/len(y_test)


def test_with_reference(hiddenstates, svm_model, mlp_model, scaler, layer_nums, data_type, top_k = 10, label = 1, target_layers=None, alpha=1, thre=None, surffix=''):

    if target_layers is None:
        target_layers = range(0, layer_nums-1)
    features_all_data = []
    labels_all_data = []
    
    
    for hiddenstate in hiddenstates:
        features_all_layer = []
        for j in target_layers:
            top_indices_all_data = []
            for data_t in data_type:
                mask_avg = np.load(f'/path/to/mask_avg')
                if thre is None:
                    if top_k:
                        topk_indices = np.argsort(mask_avg)[-top_k:]
                    else:
                        raise
                else:
                    topk_indices = np.where(mask_avg>=thre)[0]
                top_indices_all_data.append(topk_indices)
            top_indices_all_data = np.unique(np.concatenate(top_indices_all_data, axis=0))
            features_ae, labels_ae = process_data(hiddenstate, top_indices_all_data, j,)
            features_all_layer.append(features_ae)
        labels_all_data.append(labels_ae)
        features_all_layer = np.concatenate(features_all_layer, axis=1)
        features_all_data.append(features_all_layer)
    features_all_data = np.concatenate(features_all_data, axis=0)
    labels_all_data = np.concatenate(labels_all_data, axis=0)
    
    X_test, y_test  = features_all_data, labels_all_data, 
    
    results = []
    if svm_model:
        svm_test = svm_model.predict(X_test)  
        svm_test_acc = sum(svm_test == label)/len(y_test)
        results.append(svm_test_acc)
    if mlp_model:
        X_test_scaled = scaler.transform(X_test)
        mlp_test = mlp_model.predict(X_test_scaled)
        mlp_test_acc = sum(mlp_test == label)/len(y_test)
        results.append(mlp_test_acc)
    
    return results



device = torch.device("cuda:0")


# Step 2: Create the model
random_number = random.randint(1, 2000)
random.seed(random_number)
np.random.seed(random_number)
torch.manual_seed(random_number)

model_name = ''
train_num = 1
attack_mode = "PGD"
attack_power = 16
attack_iters = 1000
# round = "A"
normal_inputs = get_data("./exp_data/normal_prompt.csv")

outputs_set = {}
model_name = 'llava'

root = 'hiddenstates/'
hiddenstate_jamllm = torch.load(root + f"forward_info_jamllm_{model_name}.pt")
hiddenstate_normal = torch.load(root + f'forward_info_normal_{model_name}.pt')
hiddenstate_FS = torch.load(root + f"forward_info_FS_{model_name}.pt")
hiddenstate_JBV = torch.load(root + f"forward_info_JBV_{model_name}.pt")

hiddenstate_mmbench = torch.load(root + f"forward_info_mmbench_{model_name}.pt")
hiddenstate_mmvet = torch.load(root + f"forward_info_mmvet_{model_name}.pt")

layer_nums = len(hiddenstate_jamllm[0]['hidden_states'])
attack_type = 'jbv'
surffix = '_sorry'
target_layer_all = list(range(25))

topk = 3
alpha = 1
thre = None

max_jamllm = 0
max_mmbench = 0
max_fig = 0
max_mmvet = 0
max_normal = 0
max_JBV = 0

results_k_t = []

for thre in [   0.3, 0.4, 0.5, 0.6 ]:
    y_fig = []
    y_jamllm = []
    y_jbv = []
    y_mmbench = []
    y_mmvet = []
    y_normal = []
    for k in range(1, len(target_layer_all)+1):
        target_layer = target_layer_all[:k]
        random_items = list(hiddenstate_JBV.items())[:300]
        rand_hiddenstate = dict(random_items)
        svm_model, acc_JBV_svm, mlp_model, scaler, acc_jbv_mlp = train_with_reference([rand_hiddenstate, hiddenstate_normal],layer_nums, data_type=['jbv', ], target_layers=target_layer, alpha=alpha, top_k=topk, thre=thre, surffix=surffix)
        acc_fig_svm, acc_fig_mlp = test_with_reference([hiddenstate_FS], svm_model,  mlp_model, scaler, layer_nums , data_type=['jbv', ], label=0, target_layers=target_layer, alpha=alpha, top_k=topk, thre=thre, surffix=surffix)
        acc_jamllm_svm, acc_jamllm_mlp = test_with_reference([hiddenstate_jamllm],svm_model, mlp_model, scaler, layer_nums, data_type=['jbv', ], label=0, target_layers=target_layer, alpha=alpha, top_k=topk, thre=thre, surffix=surffix)
        acc_JBV_svm, acc_JBV_mlp = test_with_reference([hiddenstate_JBV],svm_model,  mlp_model, scaler, layer_nums, data_type=['jbv', ], label=0, target_layers=target_layer, alpha=alpha, top_k=topk, thre=thre, surffix=surffix)
        acc_mmbench_svm, acc_mmbench_mlp = test_with_reference([hiddenstate_mmbench],svm_model,  mlp_model, scaler, layer_nums, data_type=['jbv', ], label=2, target_layers=target_layer, alpha=alpha, top_k=topk, thre=thre, surffix=surffix)
        acc_mmvet_svm, acc_mmvet_mlp = test_with_reference([hiddenstate_mmvet],svm_model,  mlp_model, scaler, layer_nums, data_type=['jbv', ], label=2, target_layers=target_layer, alpha=alpha, top_k=topk, thre=thre, surffix=surffix)
        acc_normal_svm, acc_normal_mlp = test_with_reference([hiddenstate_normal],svm_model,  mlp_model, scaler, layer_nums, data_type=['jbv', ], label=2, target_layers=target_layer, alpha=alpha, top_k=topk, thre=thre, surffix=surffix)

        if acc_JBV_svm >= max_JBV:
            max_JBV = acc_JBV_svm
            JBV_state = (target_layer, thre)
        if acc_fig_svm >= max_fig:
            max_fig = acc_fig_svm
            fig_state = (target_layer, thre)
        if acc_jamllm_svm >= max_jamllm:
            max_jamllm = acc_jamllm_svm
            jamllm_state = (target_layer, thre)
        if acc_mmbench_svm >= max_mmbench:
            max_mmbench = acc_mmbench_svm
            mmbench_state = (target_layer, thre)
        if acc_mmvet_svm >= max_mmvet:
            max_mmvet = acc_mmvet_svm
            mmvet_state = (target_layer, thre)
        if acc_normal_svm >= max_normal:
            max_normal = acc_normal_svm
            normal_state = (target_layer, thre)     
        
        print(f'svm: target_layer: {len(target_layer)}, top k: {thre}, '
              f'acc_jbv: {round(acc_JBV_svm,3)}; acc_fig: {round(acc_fig_svm,3)}; ' 
                  f'acc_jamllm: {round(acc_jamllm_svm,3)}; acc_mmbench: {round(acc_mmbench_svm,3)}, '
                     f'acc_mmvet: {round(acc_mmvet_svm,3)}, acc_normal: {round(acc_normal_svm,3)}')
        
        print(f'mlp: target_layer: {len(target_layer)}, top k: {thre}, '
              f'acc_jbv: {round(acc_JBV_mlp,3)}; acc_fig: {round(acc_fig_mlp,3)}; ' 
                  f'acc_jamllm: {round(acc_jamllm_mlp,3)}; acc_mmbench: {round(acc_mmbench_mlp,3)}, '
                     f'acc_mmvet: {round(acc_mmvet_mlp,3)}, acc_normal: {round(acc_normal_mlp,3)}')
    

print('JBV:', max_JBV, JBV_state)
print('Fig:', max_fig, fig_state)
print('jamllm:', max_jamllm, jamllm_state)
print('mmvet:', max_mmvet, mmvet_state)
print('mmbench:', max_mmbench, mmbench_state)
print('normal:', max_normal, normal_state)
