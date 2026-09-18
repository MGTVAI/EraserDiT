<h1 align="center">
  <span style="color:#2196f3;"><b>EraserDiT</b></span>: Fast Video Inpainting with Diffusion Transformer Model
</h1>

<p align="center">
  <a href="https://huggingface.co/jieeliu/EraserDiT"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/JieLiu95/EraserDiT"><img alt="Github" src="https://img.shields.io/badge/EraserDiT-github-black"></a>
  <a href="https://arxiv.org/abs/2506.12853"><img alt="arXiv" src="https://img.shields.io/badge/EraserDiT-arXiv-b31b1b"></a>
  <a href="https://jieliu95.github.io/EraserDiT_demo/"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</p>

---

## 🗺️ Open-Source Roadmap

### 🛠️ In Progress
- [ ] Gradio demo
- [ ] Multi-GPU inference support

### ✅ Completed
- [x] Single-GPU inference
- [x] Model weights release
- [x] Paper publication

---

## 🚀 Overview

**EraserDiT:** Interactively removes specified objects and automatically generates the corresponding prompts. It processes a 2K‑resolution video (2160×2100, 97 frames) in only 65 seconds on a single NVIDIA H800 GPU without any acceleration. Experiments show strong performance in content fidelity, texture restoration, and temporal consistency.


---
## 🎯 Install dependencies

```
pip install -r requirements.txt
```

The pinned environment uses Python 3.10, Torch 2.6.0/cu126, and
FlashAttention 2.8.3. Create a fresh environment and install Torch first so
FlashAttention can build against the already-installed Torch:

```
conda create -n EraserDiT python=3.10 -y
conda activate EraserDiT
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 triton==3.2.0 \
  --extra-index-url https://download.pytorch.org/whl/cu126
CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH \
  MAX_JOBS=8 pip install --no-build-isolation -r requirements.txt
```
---
## 🧸 Inference
EraserDiT requires >60GB GPU memory for a 2K‑resolution video. 
Multi‑GPU support is in progress and will be open‑sourced later.
```
export HF_ENDPOINT=https://hf-mirror.com
conda activate EraserDiT
CUDA_VISIBLE_DEVICES=3 python3 inference.py --vid_path data/10268234.mp4 --mask_path data/10268234_mask.mp4 --prompt "There is a bridge over the lake." 
```
---
## 📜 Citation

If you find our work helpful, please consider giving a star 🌟 and citation 📝

```
@article{liu2025eraserdit,
  title={EraserDiT: Fast Video Inpainting with Diffusion Transformer Model},
  author={Liu, Jie and Hui, Zheng},
  journal={arXiv preprint arXiv:2506.12853},
  year={2025}
}
```
