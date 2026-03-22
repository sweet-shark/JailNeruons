# From “Sure” to “Sorry”: Detecting Jailbreak in Large Vision Language Models via JailNeurons

Official code accompanying the ICLR 2026 paper **“From ‘Sure’ to ‘Sorry’: Detecting Jailbreak in Large Vision Language Model via JailNeurons.”** This repository implements **JailNeuron** localization (gradient-based masking on internal layers to steer the model toward a refusal token such as “Sorry”) and downstream steps to aggregate masks, extract hidden states, and train lightweight detectors on selected neuron activations.

---

## Overview

Multimodal jailbreaks can elicit harmful outputs from vision-language models. This codebase:

1. **Scores jailbreak success** on a multimodal jailbreak dataset (e.g., JailBreakV) and saves which samples bypass refusal-style prefixes.
2. **Trains per-layer intervention masks** (`train_mask`) so that, when applied at a chosen transformer layer, the next-token distribution is pushed toward a **“Sorry”** (or equivalent refusal) token—surfacing neuron dimensions that matter for that behavior (**JailNeurons**).
3. **Aggregates masks** across successful jailbreak examples per layer and optionally **compares layer-wise neuron counts** above a threshold.
4. **Caches forward hidden states** for jailbreak vs. benign benchmarks (e.g., MM-Vet) for classifier training.
5. **Trains/evaluates detectors** (e.g., One-class SVM, linear SVM + MLP in `multilayer_classifier.py`) on activations restricted to JailNeuron indices.

---

## Repository layout

| Path | Role |
|------|------|
| `MLLM_models.py` | Wrappers for **LLaVA**, **Qwen-VL**, **Janus**, and **MiniGPT-4** with shared utilities and `train_mask()` for JailNeuron mask optimization. |
| `mask_jbv.py` | Runs mask training on **successful jailbreak** indices from JailBreakV; writes per-sample, per-layer mask `.pt` files. |
| `analyze_mask.py` | Loads masks, applies sigmoid, **averages** across samples per layer; saves `*_avg_*.npy`. |
| `compare_avg_mask.py` | Summarizes how many neuron indices exceed a mask threshold per layer (layer ranking / selection). |
| `v1_mprompt_explanation_jbv.py` | Iterates jailbreak samples and saves **hidden-state caches** for detection training (`forward_info_*`). |
| `v1_mprompt_explanation_mmvet.py` | Same for **MM-Vet** (benign / benchmark side). Requires `mm-vet` JSON + images. |
| `multilayer_classifier.py` | Builds features from cached hidden states using **top-k** or thresholded JailNeuron indices; trains **One-class SVM**, **SVC**, **MLP** (expects precomputed mask `.npy` paths and helper modules—see below). |
| `attack/attack_jbv.py` | Runs full inference on JailBreakV, labels jailbreaks vs. refusals via prefix list, saves `attack_success.pt`. |
| `attack/v1_mprompt_batch_llava.py` | **Image-space PGD** attack on LLaVA (`my_pgd.py`) for related adversarial experiments. |
| `attack/my_pgd.py` | PGD utilities used by the batch attack script. |
| `dataset/advbench/` | Example harmful-behavior text pairs (AdvBench-style CSV) for reference or other experiments. |

---

## Requirements

- **Python 3** and **PyTorch** with CUDA (scripts assume GPU; paths use `cuda`).
- **Hugging Face `transformers`**, **`qwen_vl_utils`** (for Qwen-VL), **`tqdm`**, **`pandas`**, **`numpy`**, **`scikit-learn`**, **`matplotlib`** (where used).
- **Model-specific assets** (checkpoints, configs) for whichever backbone you enable in `MLLM_models.py`.

`MLLM_models.py` adds local paths for **Janus** and **MiniGPT-4**-style evaluation:

```python
sys.path.append('../../model_framework/Janus')
sys.path.append('../../model_framework')
```

Adjust these to your checkout of those projects, or install/configure equivalents. MiniGPT-4 also expects a YAML config path (see `parse_args()` defaults).

---

## Data

- **JailBreakV (28K)**: multimodal jailbreak benchmark; scripts expect a CSV such as `mini_JailBreakV_28K.csv` / `JailBreakV_28K.csv` with columns like `jailbreak_query` and `image_path`, plus images under a root you pass in code or CLI.
- **MM-Vet**: download the benchmark; set `mmvet_path` to the folder containing `images/` and `mm-vet.json` in `v1_mprompt_explanation_mmvet.py`.

Replace every placeholder such as `/path/to/your/JailBreakV_28k` or `/path/to/your/jailbreakv` with your local paths before running.

---

## Typical workflow

1. **Jailbreak labeling (optional if you already have `attack_success.pt`)**  
   ```bash
   python attack/attack_jbv.py --root /path/to/JailBreakV_28k --save_path /path/to/output
   ```  
   Edit `attack_jbv.py` to select the target class (e.g., `Qwen_vl()` vs. `LLaVA()`).

2. **JailNeuron masks per successful sample**  
   Configure `model_name`, `layer_idxs`, paths, and output dir in `mask_jbv.py`, then run:
   ```bash
   python mask_jbv.py
   ```

3. **Aggregate masks per layer**  
   ```bash
   python analyze_mask.py
   ```  
   Produces averaged mask arrays used to define JailNeuron index sets.

4. **Inspect layer-wise neuron counts (optional)**  
   ```bash
   python compare_avg_mask.py
   ```

5. **Cache hidden states**  
   Run `v1_mprompt_explanation_jbv.py` and/or `v1_mprompt_explanation_mmvet.py` after pointing to your data and model. These depend on **`v1_expalanation_utils`** (and `init_exp`, `get_forward_info`, etc.) from your full project layout—ensure those modules are on `PYTHONPATH` or co-located as in the original experiment.

6. **Train detectors**  
   Configure `multilayer_classifier.py`: it imports **`load_data`** and **`v1_expalanation_utils`**, and uses hardcoded mask paths under `inv_mask/...` in the published snippet—**align paths and dependencies** with your machine before running.

---

## Configuration notes

- **Model switch**: Instantiate the desired class in each entry script (`LLaVA`, `Qwen_vl`, `Janus`, `MiniGPT` in `MLLM_models.py`).
- **“Sorry” token id**: `mask_jbv.py` uses model-specific tokenizer encodings for the refusal string; keep these consistent with your checkpoint.
- **Layers**: `layer_idxs` in `mask_jbv.py` and loops in `analyze_mask.py` / `compare_avg_mask.py` should match the backbone depth you use.
- **Suffix**: Scripts use e.g. `suffix = '_sorry'` so outputs do not overwrite other runs.

---

## Limitations and research use

This code is released for **research on safety and robustness** of vision-language models. Many files contain **absolute paths and missing local helpers** from the authors’ cluster layout; expect to adapt imports, paths, and optional dependencies before full reproduction.

Do not use this software to develop or deploy harmful applications. Follow your institution’s policies and applicable laws when working with jailbreak or harmful-content datasets.

---

## Citation

If you use this code, please cite the ICLR 2026 paper (title as above). Use the BibTeX entry from the camera-ready PDF when available.

