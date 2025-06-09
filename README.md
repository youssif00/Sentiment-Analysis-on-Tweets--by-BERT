# Emotion Detection from Tweets BY BERT

A deep learning project that performs sentiment analysis on tweets using the BERT (Bidirectional Encoder Representations from Transformers) model. This project classifies tweets into 6 different emotions: sadness, joy, love, anger, fear, and surprise.

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

