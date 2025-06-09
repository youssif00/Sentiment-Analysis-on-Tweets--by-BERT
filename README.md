# Emotion Detection from Tweets BY BERT

An AI and  deep learning project that performs sentiment analysis on tweets using the BERT (Bidirectional Encoder Representations from Transformers) model. This project classifies tweets into 6 different emotions: sadness, joy, love, anger, fear, and surprise.

## Overview

This project uses the BERT model to perform multi-class sentiment analysis on tweets. The model is fine-tuned on the DAIR-AI emotion dataset to classify tweets into six different emotional categories. The implementation uses the Hugging Face `transformers` library and PyTorch.

## Features

- Multi-class emotion classification for tweets
- Fine-tuned BERT model
- Data visualization of emotion distribution
- High accuracy on emotion classification
- Support for GPU acceleration using mixed precision training

## Dataset

The project uses the DAIR-AI emotion dataset which contains:
- Training set: 16,000 tweets
- Validation set: 2,000 tweets
- Test set: 2,000 tweets
The dataset is labeled with 6 emotion categories:
- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## Requirements

```
transformers
datasets
pandas
seaborn
matplotlib
wordcloud 
scikit-learn
torch
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/youssif00/Sentiment-Analysis-on-Tweets--by-BERT.git
cd Sentiment-Analysis-on-Tweets--by-BERT
```

2. Install the required packages:
```bash
pip install transformers datasets pandas seaborn matplotlib wordcloud scikit-learn torch
```
## Usage

1. Open the Jupyter notebook `project_bert_we.ipynb`
2. Run the cells sequentially to:
   - Load and preprocess the data
   - Visualize emotion distribution
   - Fine-tune the BERT model
   - Train and evaluate the model

## Model Architecture

The project uses the `bert-base-uncased` model from Hugging Face's transformers library with:
- 12 transformer layers
- Hidden size of 768
- 12 attention heads
- Fine-tuned classification head for 6 emotion classes
## Training

The model is trained with the following configuration:
- Number of epochs: 3
- Batch size: 16 (effective batch size 32 with gradient accumulation)
- Mixed precision training (FP16)
- AdamW optimizer
- Learning rate scheduler

## Results

The model achieves high accuracy in classifying emotions from tweets. Detailed performance metrics and visualizations can be found in the notebook.
## License

[Include license information here]

## Contributors

- [youssif00](https://github.com/youssif00)

## Acknowledgments

- Hugging Face for the transformers library
- DAIR-AI for the emotion dataset
- The BERT team at Google Research

