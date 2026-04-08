<div align="center">

# 🧠 ConvED-SR

**Enhancing SEEG-based Speech Decoding via Convolutional Encoder-Decoder and Scale-Recursive Reconstructor**

[![Paper](https://img.shields.io/badge/Paper-IEEE_Sensors_Journal-blue.svg?style=for-the-badge&logo=ieee)](https://ieeexplore.ieee.org/document/10980144)
[![Python](https://img.shields.io/badge/Language-Python-3776AB.svg?style=for-the-badge&logo=python)](https://github.com/yuanyu-1016/ConvED-SR)
[![License](https://img.shields.io/github/license/yuanyu-1016/ConvED-SR?style=for-the-badge&color=green)](https://github.com/yuanyu-1016/ConvED-SR/blob/main/LICENSE)

[📖 Paper (IEEE)](#-publication) • [🎯 Overview](#-overview) • [📂 Dataset & Audio Samples](#-audio-samples) • [⚙️ Getting Started](#️-getting-started) • [🙏 Citation](#-citation)

</div>

---

## 📖 Publication

This repository contains the official PyTorch implementation for the paper:

> **Enhancing SEEG-based Speech Decoding via Convolutional Encoder-Decoder and Scale-Recursive Reconstructor** > *Published in: IEEE Sensors Journal* > **Link to Paper**: [https://ieeexplore.ieee.org/document/10980144](https://ieeexplore.ieee.org/document/10980144)

---

## 🎯 Overview

Decoding speech directly from brain activity is a significant challenge in brain-computer interfaces (BCIs). **ConvED-SR** introduces a novel deep learning framework to synthesize high-quality audio directly from Stereoelectroencephalography (sEEG) signals.

### Model Architecture
The proposed architecture addresses the complex spatial-temporal features of sEEG data through two main components:
1. **Convolutional Encoder-Decoder (ConvED)**: Effectively captures complex neural representations from raw sEEG inputs.
2. **Scale-Recursive Reconstructor (SR)**: Progressively reconstructs speech waveforms or spectrograms across different scales, ensuring high fidelity and temporal coherence in the decoded speech.

*(You can view the detailed architecture diagram in `ConvED-SR.png` included in this repository).*

---

## 📂 Repository Structure

```text
ConvED-SR/
├── audio_samples/      # Reconstructed speech audio samples comparing our method with baselines
├── src/                # Core Python source code (PyTorch)
│   ├── model.py        # Implementation of ConvED and Scale-Recursive Reconstructor
│   ├── data_loader.py  # sEEG data preprocessing and PyTorch Datasets
│   ├── train.py        # Training scripts
│   └── evaluate.py     # Evaluation and metric calculation (PESQ, STOI, etc.)
├── ConvED-SR.png       # High-res diagram of the proposed model architecture
├── sub-03_fold_0.png   # Evaluation visualization example (e.g., spectrogram comparison)
└── README.md           # This file
```

### 🎧 Audio Samples
To evaluate the subjective quality of the decoded speech, we have provided reconstructed audio samples in the `audio_samples/` directory. These files demonstrate the superior auditory reconstruction capabilities of ConvED-SR compared to baseline methods.

---

## ⚙️ Getting Started

### 1. Prerequisites
The code is written in Python. Ensure you have a working environment with PyTorch installed. It is recommended to use a virtual environment or Conda.

```bash
# Clone the repository
git clone [https://github.com/yuanyu-1016/ConvED-SR.git](https://github.com/yuanyu-1016/ConvED-SR.git)
cd ConvED-SR

# Install dependencies (Assuming a requirements.txt is added later, or install manually)
pip install torch torchaudio numpy scipy matplotlib
```

### 2. Data Preparation
*(Please add instructions here regarding how to request or format the sEEG dataset, as medical data is usually kept private or requires specific preprocessing steps).*

### 3. Training the Model
To train the ConvED-SR model from scratch, navigate to the `src` directory and run the training script. Example command:
```bash
python src/train.py --config configs/default.yaml
```

### 4. Evaluation and Inference
To decode speech from sEEG test data using a pre-trained model and generate predicted spectrograms/waveforms:
```bash
python src/evaluate.py --checkpoint path/to/weights.pth
```

---

## 🙏 Citation

If you find this code or our paper useful in your research, please consider citing our work:

```bibtex
@article{he2025convedsr,
  author={He, Yuanyu and others},
  journal={IEEE Sensors Journal}, 
  title={Enhancing SEEG-based Speech Decoding via Convolutional Encoder-Decoder and Scale-Recursive Reconstructor}, 
  year={2025},
  volume={},
  number={},
  pages={},
  doi={10.1109/JSEN.2025.XXXXXXX} 
}
```
*(Note: Please update the volume, issue, pages, and exact DOI details once they are fully indexed on IEEE Xplore).*

---

## ✉️ Contact
For any questions regarding the code or the paper, please open an issue in this repository or contact the first author (**Yuanyu He**) directly via email.
