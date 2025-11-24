# Preprocessing-Dataset-NLP
LM-Powered SMS Spam Detection

Project Overview

This project demonstrates how to fine-tune a small, pre-trained Transformer-based Language Model (LLM) for a binary text classification task: identifying whether a Short Message Service (SMS) text is HAM (legitimate) or SPAM (unsolicited/malicious).

The goal was to achieve high classification performance using resource-efficient settings, making the training process quick and feasible on a CPU-only environment.

Key Technologies Used

Model: prajjwal1/bert-tiny (A highly compressed version of BERT for fast training).

Dataset: ucirvine/sms_spam (The classic SMS Spam Collection dataset).

Framework: Hugging Face transformers and datasets.

Deployment (Optional): Streamlit.

🚀 Getting Started

1. Prerequisites

Ensure you have Python 3.8+ and a virtual environment activated.

2. Installation

Install all required libraries, including transformers, datasets, and streamlit for the optional deployment.

pip install transformers datasets accelerate -U
pip install evaluate scikit-learn numpy pandas
pip install streamlit


3. Running the Fine-Tuning

The entire fine-tuning process is contained in the Jupyter Notebook: Fine-Tuning Spam Detection.ipynb.

Open the Notebook: Launch your Jupyter environment and open the notebook.

Run Cells Sequentially: Execute all cells from top to bottom.

The notebook performs the following steps:

Loads the ucirvine/sms_spam dataset.

Splits the data into Train (80%), Validation (10%), and Test (10%).

Tokenizes the messages using the bert-tiny tokenizer.

Fine-tunes the model using the Hugging Face Trainer API with the following hyper-parameters optimized for speed:

Model: prajjwal1/bert-tiny

Epochs: 2

Batch Size: 32

Evaluates the final model on the test set, calculating Accuracy, F1-Score, Precision, and Recall.

Generates and displays a Confusion Matrix.

💾 Model Artifacts

After successful training, the fine-tuned model and its tokenizer are saved to the local directory:

Model Path: ./fine_tuned_spam_detector

This directory contains the necessary files (pytorch_model.bin, config.json, tokenizer.json, etc.) for direct inference using the Hugging Face pipeline.

🌐 Optional Deployment (Streamlit)

A simple web interface using Streamlit is provided to interact with the trained model.

1. Save the Deployment File

Ensure the inference and deployment code is saved in a separate file named app.py in the same directory as the ./fine_tuned_spam_detector folder.


📈 Evaluation Results (Expected Metrics)

Due to the use of the compressed bert-tiny model for faster CPU training, the model achieves impressive speed with only a minor trade-off in accuracy compared to larger models.

Metric

Target Goal

Accuracy

$> 98\%$

F1-Score (Macro)

$> 0.95$

Recall (SPAM)

High (Low False Negatives)

The Confusion Matrix provides a crucial visual assessment of the model's performance, particularly showing its ability to minimize False Negatives (failing to catch actual spam).
