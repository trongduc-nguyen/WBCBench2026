# WBCBench 2026: White Blood Cell Classification
**PathMedAI Team – ISBI 2026 Grand Challenge**

This repository contains the reproducibility package for our **white blood cell classification solution** submitted to WBCBench 2026.

**Challenge Link:** [WBCBench 2026](https://www.kaggle.com/competitions/wbc-bench-2026)  
**Accepted Paper:** [ISBI 2026 Paper](https://arxiv.org/abs/2603.16249)

---

## Table of Contents
- [Environment Requirements](#environment-requirements)
- [Data Preparation (End-to-End)](#data-preparation-end-to-end)
- [Training Pipeline](#training-pipeline)
- [Inference & Submission](#inference--submission)
- [Contact](#contact)

---

## Environment Requirements
- **OS:** Ubuntu 22.04.4 LTS  
- **Python:** 3.10  
- **Network:** Required to download pre-trained weights (Swin-T, CellPose via timm/huggingface)  

**Setup Environment:**
```bash
conda create -n wbc python=3.10 -y
conda activate wbc
pip install -r requirements.txt
````

---


---

## Data Preparation (End-to-End)

1. Extract raw Kaggle data:

```bash
unzip wbc-bench-2026.zip -d dataset_wbc/
```

2. Preprocessing & restoration:

```bash
python prepare_wbc_dataset_cellpose.py
python split_dataset_by_noise.py
python gen_data_separate_AB.py
```

3. Train Pix2Pix GAN for denoising:

```bash
cd pytorch-CycleGAN-and-pix2pix
python datasets/combine_A_and_B.py
python train.py --dataroot ./datasets/wbc_denoising_final --name wbc_denoise --model pix2pix --direction AtoB --batch_size 32 --preprocess none
```

4. Generate restored images:

```bash
python inference_gan_restore.py
```

5. Final dataset assembly:

```bash
cd ..
python merge_datasets.py
python crop.py
```

---

## Training Pipeline

1. Train Swin Transformer ensemble (5 folds):

```bash
python train_swin.py
```

2. Generate embeddings & train contrastive head:

```bash
python generate_embedding.py
python train_contrastive_head.py
```

---

## Inference & Submission

```bash
python submit.py
# Output: submission.csv
```

> **Output:** `submission.csv` ready for Kaggle submission.

---

## Contact

We welcome feedback or reports of any technical issues.

**Team Name:** PathMedAI

**Leader Contact:** Nguyen Trong Duc

**Email:** [nggtduc@gmail.com](mailto:nggtduc@gmail.com)

## Acknowledgements

We would like to sincerely thank the developers and contributors of the **Pix2Pix GAN** and **CycleGAN** projects for providing high-quality, open-source implementations.  
Their work made it possible for us to implement the **image restoration pipeline** efficiently, improving the quality of white blood cell images for downstream classification.  

Special thanks to the open-source community whose models, tools, and datasets enabled rapid experimentation and reproducibility in our WBCBench 2026 submission.
