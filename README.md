# Emotion-detection-by-BERT
Background: Social media users express a wide range of emotions in short messages. Automatically detecting these emotions can benefit applications in customer support, mental health monitoring, and more.

This repository contains a project that builds and analyzes a multi-class emotion classifier using a BERT-based transformer model on a dataset of tweets labeled with six emotions.

able of Contents

Project Overview

Dataset

Setup & Installation

Usage

Project Structure

Results

Future Work

License

Project Overview

Social media users express a variety of emotions in short messages. Accurately detecting these emotions enables applications in customer support, mental health monitoring, and more. In this project, we fine-tune a BERT-based model (bert-base-uncased) to classify tweets into one of six emotions: sadness, joy, love, anger, fear, and surprise.

Key components:

Data loading and visualization (class distribution, sample tweets, word clouds).

Preprocessing with dynamic padding.

Fine-tuning using Hugging Face Trainer with mixed precision (FP16).

Performance evaluation and analysis.

Dataset

We use the dair-ai/emotion dataset from Hugging Face, which contains approximately 20,000 English tweets labeled with six emotion categories.

Setup & Installation

Clone this repository

git clone https://github.com/<your-username>/emotion-detection-tweets.git
cd emotion-detection-tweets

Create a virtual environment (optional but recommended)

python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

requirements.txt should include:

torch
transformers
datasets
pandas
matplotlib
seaborn
wordcloud
scikit-learn


Usage

1. Data Visualization

Run the visualize.py script to explore the dataset:

python visualize.py

This will display:

Class distribution bar chart

Sample tweets per emotion

Word clouds for each emotion

2. Training

Fine-tune the BERT model using:

python train.py --epochs 3 --batch_size 16 --fp16

Options:

--epochs: Number of training epochs (default: 3)

--batch_size: Per-device batch size (default: 16)

--fp16: Enable mixed-precision training

3. Evaluation

Evaluate on the test set:

python evaluate.py --checkpoint ./results/checkpoint-best

4. Prediction

Use predict.py to classify a custom tweet:

python predict.py --text "I'm so excited for today!"

Project Structure

emotion-detection-tweets/
├── data/                  # (optional) pre-downloaded data
├── visualize.py           # Data exploration and visualization
├── train.py               # Model fine-tuning script
├── evaluate.py            # Evaluation script
├── predict.py             # Single-sentence prediction script
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── results/               # Training outputs and checkpoints

Results

Test Accuracy: ~94%
