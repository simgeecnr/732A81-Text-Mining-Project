# Comparative Analysis of DistilBERT and Traditional Classifiers for Multi-Class News Classification

This repository contains the implementation of a **text classification project** comparing **traditional machine learning methods** (Naive Bayes and Logistic Regression) with an **advanced pre-trained transformer model**, DistilBERT. The project uses the **AG News dataset** and evaluates model performance under varying dataset sizes to study how data availability impacts classifier accuracy.

---

## Project Overview

Text classification is a key task in **Natural Language Processing (NLP)**, aiming to categorize textual data into predefined categories. This project:

- Compares **Naive Bayes** and **Logistic Regression** against **DistilBERT**.  
- Performs **hyperparameter tuning** on DistilBERT, focusing on hidden size, dropout rate, and number of epochs.  
- Uses **training subsamples of 400, 1000, 5000, and 20,000 examples** to evaluate model performance across dataset sizes.  

**Key finding:** DistilBERT consistently outperforms traditional ML models, especially on **smaller datasets**, highlighting the benefits of pre-trained models when training data is limited.

---

## Key Files

- `code/model_400.ipynb` – models trained with 400 samples  
- `code/model_1000.ipynb` – models trained with 1,000 samples  
- `code/model_5000.ipynb` – models trained with 5,000 samples  
- `code/model_20000.ipynb` – models trained with 20,000 samples  
- `code/full_baseline.ipynb` – Naive Bayes and Logistic Regression models trained on the full dataset  

---

## Dataset

The AG News dataset is used in this project and can be accessed from Kaggle:  
[AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

---

## Technologies Used

- **Programming & ML Frameworks:** Python, PyTorch, Hugging Face Transformers  
- **Data Handling & Analysis:** pandas, NumPy  
- **NLP:** Pre-trained Transformer Models, Tokenization, Text Classification  

