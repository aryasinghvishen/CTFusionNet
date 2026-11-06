# CTFusionNet

**A lightweight hybrid CNN–Transformer with bidirectional cross-task fusion for simultaneous low-dose CT denoising and lesion localization**

Arya Singh Vishen, Dr. Saravanan M, Dr. Nivedhitha M
Department of Networking and Communications, School of Computing, SRM Institute of Science and Technology, Kattankulathur, Chennai, India

---

CTFusionNet takes a single low-dose CT slice and, in one forward pass, returns:

1. a **denoised** version of the slice, and
2. **bounding boxes** around the lesions it finds.

The two tasks share one encoder, and their decoders exchange information through a **bidirectional cross-task fusion (CTF)** module:

- **Localization → denoising:** a lesion-attention map tells the denoiser where detail must be preserved.
- **Denoising → localization:** channel weights pass noise-suppressed evidence to the localizer.

Each direction has a learnable gate that starts at zero. Training therefore begins from the unfused network, and the learned gate values show how much each direction is actually used.

![Architecture](figures/architecture.png)

![Cross-task fusion module](figures/ctf_module.png)

## What is new

Earlier work either denoises low-dose CT without regard to the diagnostic task (RED-CNN, CTformer, Hformer), or links a denoiser and a *separate* task network through a training loss only (LIDnet, TOD-Net). CTFusionNet places both tasks in **one network** and lets them exchange **features in both directions**, with gates that make the use of each direction measurable.

## Results

Evaluated on the official DeepLesion test split: 1,096 slices with 1,124 lesions, using low-dose inputs simulated at I₀ = 10⁶ (low) and 3×10⁶ (moderate). The comparison is with the same network trained without CTF (same data, schedule and seed).

| Low dose | Low-dose input | No fusion | **CTFusionNet** |
|---|---|---|---|
| PSNR (dB)* | 41.15 | **46.98** | 44.67 |
| SSIM | 0.901 | **0.970** | 0.959 |
| RMSE in body (HU) | 50.6 | **29.6** | 38.7 |
| Sensitivity @ 4 FP/image | – | 0.508 | **0.573** |
| Mean FROC sensitivity (0.5–8 FP) | – | 0.436 | **0.475** |
| Parameters | – | 3.17 M | 3.43 M |
| Time per 512×512 slice (T4) | – | 32.6 ms | 32.3 ms |

\*PSNR/SSIM are computed over the full [−1024, 3071] HU range, so the absolute values are higher than in papers that use a narrow display window. Compare values within this table only.

- **Fusion improves lesion localization:** +3.8 points mean sensitivity (bootstrap 95% CI [0.017, 0.058]) and +6.5 points at 4 FP/image. The largest gains are for pelvic, abdominal and mediastinal lesions.
- **Fusion lowers denoising fidelity by 2.3 dB PSNR.** About half of this comes from checkpoint selection (see the paper).
- The 1/4-scale localization→denoising gate grows from 0 to 0.96 during training.

## Dataset

[NIH DeepLesion](https://nihcc.app.box.com/v/DeepLesion) (Yan *et al.*, J. Med. Imaging, 2018). The notebook downloads the image archives directly from NIH and checks their MD5 checksums. We used the first **10 of 56 archives**: 6,971 key slices of 512×512 from 583 patients, with the official patient-level split. **The data are not included in this repository.** Please obtain them from NIH under their terms of use.

| Split | Lesions | Slices | Patients |
|---|---|---|---|
| Train | 5,281 | 5,158 | 414 |
| Validation | 726 | 717 | 80 |
| Test | 1,124 | 1,096 | 89 |

**Low-dose simulation:** HU → attenuation → 2-D parallel-beam Radon transform (360 views) → Poisson photon noise at incident count I₀ → filtered back projection. A GPU implementation is included and checked against scikit-image (noise agreement within 1%, see `results/gpu_sim_check.csv`).

## Repository structure

```
notebooks/
  CTFusionNet_DeepLesion_Kaggle.ipynb   # full pipeline; runs on Kaggle or Colab
  CTFusionNet_DeepLesion_Colab.ipynb    # Colab-only version (Google Drive storage)
src/
  ctfusion.py   # model, CTF module, low-dose simulation, losses, decoding, metrics
  app.py        # Streamlit demo
results/        # test metrics, per-lesion-type sensitivity, significance tests, dataset stats
figures/        # figures used in the paper
```

## How to reproduce

**Kaggle (recommended for long training):**
1. Upload `notebooks/CTFusionNet_DeepLesion_Kaggle.ipynb`. Set Accelerator to **GPU T4** and turn **Internet on**.
2. **Save Version → Save & Run All (Commit).** The run downloads the data, generates the low-dose images, trains, evaluates and saves everything to the version's Output (about 7–9 hours on a T4).
3. For the comparison model, add the previous version's output as input. In the config cell, set `EXPERIMENT = 'no_fusion'` and `USE_FUSION = False`, then commit again. Data and existing checkpoints are reused.

**Colab:** open `notebooks/CTFusionNet_DeepLesion_Colab.ipynb`, select a T4 GPU and choose *Run all*. Data and checkpoints are kept in Google Drive. Free Colab sessions stop when the tab closes, but training resumes from the last epoch.

Main settings: AdamW, learning rate 3×10⁻⁴, weight decay 10⁻⁴, one-cycle schedule, batch size 8, 40 epochs, mixed precision, seed 42.

## Demo

Section 10 of the notebook starts a Streamlit app (`src/app.py`) and prints a public link. In the app you can:
- pick a DeepLesion test slice or upload one;
- choose a dose level;
- compare the low-dose input, the denoised output and the reference;
- see predicted vs. ground-truth boxes, PSNR/SSIM and the fusion attention maps.

To run it locally with a trained checkpoint:

```bash
pip install -r requirements.txt
cd src
CT_PROJECT=/path/to/CT_Multitask_Research CT_RUN=ctfusion_full_s42 streamlit run app.py
```

Trained weights (`best.pt`) are not stored in the repository. «Attach them to a GitHub Release and add the link here.»

## Limitations

- **Simulated low dose:** 2-D parallel-beam geometry with Poisson noise only, not real low-dose acquisitions.
- **Partial dataset:** 10 of 56 DeepLesion archives and 2-D key slices only.
- **Single training seed;** the localization branch overfits after about 6–8 epochs.
- **Missing comparisons:** no ablation of the self-attention blocks and no comparison with external methods (e.g. RED-CNN).

This is a research prototype and **not intended for clinical use.**

## Citation

If you use this code, please cite:

```bibtex
@misc{vishen2026ctfusionnet,
  title  = {CTFusionNet: A Lightweight Hybrid CNN--Transformer with Bidirectional Cross-Task Fusion
            for Simultaneous Low-Dose CT Denoising and Lesion Localization},
  author = {Vishen, Arya Singh and M, Saravanan and M, Nivedhitha},
  year   = {2026},
  note   = {SRM Institute of Science and Technology}
}
```

Please also cite the DeepLesion dataset:

```bibtex
@article{yan2018deeplesion,
  title   = {DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning},
  author  = {Yan, Ke and Wang, Xiaosong and Lu, Le and Summers, Ronald M.},
  journal = {Journal of Medical Imaging},
  volume  = {5}, number = {3}, pages = {036501}, year = {2018},
  doi     = {10.1117/1.JMI.5.3.036501}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The DeepLesion data are subject to the NIH Clinical Center's own terms and are not covered by this license.
