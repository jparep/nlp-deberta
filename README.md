# Fake News Detection using DeBERTa

This project implements a fake news detection model using the DeBERTa (Decoding-enhanced BERT with disentangled attention) transformer model. The project leverages a pre-trained DeBERTa model and fine-tunes it on a dataset of real and fake news articles. The goal is to classify news articles as either real or fake.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

The rise of misinformation and fake news has become a significant challenge in the digital age. Detecting fake news accurately and efficiently is crucial to ensuring the reliability of information disseminated online. This project aims to build a robust fake news detection system using the DeBERTa model, which has been fine-tuned on a dataset of real and fake news articles.

## Dataset

The project uses two CSV files:

- `real.csv`: Contains news articles labeled as real.
- `fake.csv`: Contains news articles labeled as fake.

Each CSV file should contain at least two columns:

- `text`: The content of the news article.
- `label`: The label indicating whether the news is real (`1`) or fake (`0`).

## Requirements

The project requires Python 3.7+ and the following libraries:

- `pandas`
- `torch`
- `transformers`
- `scikit-learn`

You can install the necessary packages using the following command:

```bash
pip install pandas torch transformers scikit-learn accelerate
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jparep/nlp-debertta.git
cd nlp-deberta
```

2. Install the dependencies:

```bash
    conda env create -f environment.yml
```

## Usage

1. Prepare the dataset:
    - Place the `true.csv` and  `fake.csv` files in the root directory of the project.

2. Load the data and concatenate the Dataframes


## Model Training

The model is fine-tuned using the transformers library. The training script performs the following steps:

   1.  Data Loading: Loads the real and fake news datasets.
    2. : Concatenates and shuffles the datasets, and splits them into training and testing sets.
    3. Tokenization: Uses the DeBERTa tokenizer to preprocess the text data.
    4.  Fine-Tuning: Fine-tunes the pre-trained DeBERTa model on the training data.
    5. Evaluation: Evaluates the model on the test data.


## Evaluation

The model is evaluated using the following metrics:

    - Accuracy
    - Precision
    - Recall
    - F1-Score

These metrics are printed to the console after the evaluation step.


## Saving and Loading the Model

After training, the model and tokenizer are saved to the ./fake_news_deberta_model directory. 


## Acknowledgments

   - This project uses the DeBERTa model from Microsoft, which is a state-of-the-art transformer model for natural language processing tasks.
    - The transformers library by Hugging Face provided the tools necessary to easily implement and fine-tune transformer models.


## License

This project is licensed under the MIT License - see the LICENSE file for details.