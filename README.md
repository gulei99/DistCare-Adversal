# Domain-invariant Clinical Representation Learning for Emerging Disease Prediction

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-orange.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code for the paper: **"Robust Clinical Representation Learning via Bridging Distribution Shifts across Heterogeneous EMR Datasets"**.

## Abstract

Emerging diseases pose significant challenges for symptom identification and timely clinical intervention due to limited information. This paper proposes a domain-invariant representation learning framework aimed at addressing the issues of clinical data scarcity and cross-dataset feature inconsistency in the context of emerging diseases. By constructing a **Transition Model** guided by a **Teacher Model** and incorporating **domain adversarial training**, we learn a domain-invariant feature representation. This framework effectively captures common knowledge across different medical domains and handles private features through a transfer mechanism based on **Dynamic Time Warping (DTW)**. Ultimately, it achieves superior performance across multiple emerging disease prediction tasks.

## Framework Overview

Our model adopts a three-stage transfer learning framework to enable knowledge transfer from a data-rich source domain to a data-scarce target domain.

![Model Framework](./figures/model_framework.png)
*Figure: The overall framework for the emerging disease prediction model. Left: Training the Teacher Model on the source dataset; Middle: Training the domain-invariant Transition Model; Right: Transferring knowledge to the target domain model and fine-tuning.*

1.  **Stage 1: Pretrain Teacher Model**: Train a robust Multi-Channel GRU (MCGRU) model on a large-scale source domain dataset (e.g., PhysioNet) to serve as the source of knowledge.
2.  **Stage 2: Train Transition Model**: This is the core of the framework. It accepts data from both the source and target domains simultaneously. It mimics the Teacher Model's representation via **Knowledge Distillation** and learns domain-invariant features through **Domain Adversarial Training**.
3.  **Stage 3: Transfer and Fine-tune**: Transfer the trained GRU parameters from the Transition Model to the final target model. Shared features are transferred directly, while private features are transferred by matching the most similar source domain features using **DTW** distance. Finally, the model is fine-tuned on the target domain data.

## Requirements

We recommend using conda to create an independent virtual environment.

```bash
# Create conda environment
conda create -n your_env_name python=3.7
conda activate your_env_name

# Install core dependencies
pip install torch==1.12.1 torchvision==0.13.1
pip install numpy pandas scikit-learn matplotlib

# Install DTW library
pip install dtw-python
```

## Dataset Preparation

This study utilizes the following datasets:
* **Source Domain Dataset**: [PhysioNet Challenge 2019 Sepsis Dataset](https://physionet.org/content/challenge-2019/1.0.0/)
* **Target Domain Datasets**:
    * TJH COVID-19 Dataset
    * HMH COVID-19 Dataset
    * PUTH ESRD Dataset

You need to download these datasets and perform detailed preprocessing, including data cleaning, missing value imputation, normalization, etc. Finally, please save the processed data in the format expected by the code (usually `.pkl` or `.dat` files). **Please refer to the data loading sections in each Notebook for exact filenames and path structures.**

## Usage

The experiments in this project are executed via a series of independent Jupyter Notebook (`.ipynb`) files. Each file corresponds to a specific experiment (e.g., a specific baseline model or an ablation study).

### 1. Running the Core Model (Ours / `distcare_adversal`)

To reproduce the results of the core model proposed in the paper, you need to complete the three stages of training in sequence (although they may be integrated into the same Notebook).

1.  **Stage 1: Train Teacher Model**
    *   Locate and run the script for training the teacher model (e.g., `teacher_model_train.ipynb`).
    *   This process will train on the source domain data and save the best Teacher Model checkpoint (e.g., to the `./model/` directory).

2.  **Stage 2: Train Transition Model**
    *   Locate and run the core transition model training script (e.g., `distcare_adversal_tj.ipynb`).
    *   This script will load the Teacher Model saved in Stage 1 and perform knowledge distillation and domain adversarial training using both source and target domain data.
    *   Upon completion, the best Transition Model checkpoint will be saved.

3.  **Stage 3: Fine-tune and Evaluate**
    *   Locate and run the final K-fold cross-validation script.
    *   This script will:
        *   Create the final target model (`distcare_target`).
        *   Load the Transition Model saved in Stage 2 and invoke functions like `transfer_gru_dict` to execute knowledge transfer.
        *   Perform K-fold cross-validation fine-tuning and evaluation on the target domain data.
        *   **Print the final average performance metrics** on the screen and at the end of the log file upon completion.

### 2. Running Baselines

To reproduce the results of the baseline models listed in the paper (e.g., GRU, Transformer, DANN, TimeNet, etc.), please run the corresponding Notebook files.
*   For example, to run the DANN model experiment on the TJH dataset, locate and run `dann_tj.ipynb`.
*   These scripts are usually self-contained, covering the entire process from training to evaluation for that specific model.

### 3. Plotting Convergence Curves

After running all necessary experiments and generating log files, you can use the `plot_convergence.py` script to plot the model convergence speed comparison graph.

1.  Open the `plot_convergence.py` file.
2.  In the `MODELS_TO_PLOT` dictionary, configure the log file paths (`log_dir`) and exact filenames (`filename`) for the models you wish to plot.
3.  Run `python plot_convergence.py` in the terminal.
4.  The generated image will be saved as `model_convergence_comparison.png`.

## Citing Our Work
If you use our work or code in your research, please cite our paper.
