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
---
## 🧸 Inference
EraserDiT requires >60GB GPU memory for a 2K‑resolution video. 
Multi‑GPU support is in progress and will be open‑sourced later.
```
python inference.py --vid_path data/10268234.mp4 --mask_path data/10268234_mask.mp4 --prompt "There is a bridge over the lake." 
```

### CPU offload

`--cpu_offload` keeps only the active model on the GPU: text encoder → VAE encoder →
transformer → VAE decoder. Models return to CPU after each window or on failure.
This preserves the inference operations and adds CPU memory and transfer costs.
The default remains full GPU residency.

For high-resolution inputs, `--vae_tiling` additionally enables the VAE's existing
spatial tiling to reduce activation memory. Tiling can change output values at tile
boundaries; it is separate from CPU offload and disabled by default.

```bash
HF_HUB_OFFLINE=1 uv run --no-project python inference.py \
  --model_path data/model --cpu_offload --vae_tiling \
  --vid_path data/113000356.mp4 --mask_path data/113000356_mask.mp4
```

The original code uses Diffusers internal APIs. A tested compatibility set is
Diffusers 0.33.1, Transformers 4.51.3, Tokenizers 0.21.1, Hugging Face Hub 0.30.2,
and PEFT 0.15.2 (tested with PyTorch 2.6.0/CUDA 12.6). Diffusers 0.40 is incompatible.
To use this set without replacing an existing environment:

```bash
uv pip install --target .cache/alg-baseline-deps --no-deps \
  diffusers==0.33.1 transformers==4.51.3 tokenizers==0.21.1 \
  huggingface-hub==0.30.2 peft==0.15.2
export PYTHONPATH="$PWD/.cache/alg-baseline-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

GPU integration test (local weights required):
`uv run --no-project python tests/test_stage_cpu_offload.py`.
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
