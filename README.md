# Bangla-TextClassification-EDA-and-Models
## Bangla-TextClassification-EDA-and-Models: Project Overview
- In this project i have built some models to classify (**economy,sports,international,state,technology,entertainment,education'**) using machine learning and deeplearning models.
- For dataset i have used a public dataset available called **banglamct7-bangla-multiclass-text-dataset-7-tags** which is avaiable in [this](https://github.com/user/repo/blob/branch/other_file.md) link. 
- **`Word embeding`** is used for deep learning models and **`TFIDF`** is used for machine learning models for feature represtations for extracting the semantic meaning of the words.
- Machine Learning models has been built by using a **Logistic Regression** and **Multinomial Naieve Bayes**.
- Deep learning models has been built by using a **Deep Neural Network**,**Convolutional Neural Network**,**BiDirectional LSTM** and **CNN-BiLSTM Mybrid** model.
- Finally, the models performance is evaluated using various evaluation measures such as **`confusion matrix, accuracy , precision, recall and f1-score`** with classification report.  

## Resources Used
- **Developement Envioronment :** Kaggle
- **Python Version :** 3.7
- **Framework and Packages :** Tensorflow, Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn

## Project Outline 
- Data Collection and Cleaning
- Data Summary
- Data Preparation
- Model
- Model Evaluation


## Data Collection and Cleaning
The dataset contains clean data ready to be used for feature extraction and classification.But i have applied Stopwords removal and Stemming on the clean data.

## Data Summary 
Data summary is done in a single notebook available in [this](https://github.com/NuhashHaque/Bangla-TextClassification-Analysis-EDA-and-Models/blob/main/EDA%20on%20BanglatText.ipynb) link.In EDA notebook, i have shown number of documents, words and unique words have in each category class, histogram analysis to text length and Ngram analysis upto trigram in each category.

## Data Preparation
To prepare data before model building, i have used TFIDF for machine learning and Word Embedding for deep learning models.The parameters are optimized and tuned with respect to the EDA results.

## Model
I have used Logistic Regression,Multinomial Naive Bayes,Deep Neural Networks,CNN,LSTM,CNN-BiLSTM hybrid model.
All the models parameters are tuned and optimized.


## Model Evaluation 

| Model Name  | Accuracy    | Precision     | Recall | F1-Score|
| :---        |    :----:   |   ---:        |  ---:  |  ---:   |
| Logistic Regression     | 0.9156     | 0.9157   |   0.9156    |   0.9156      |
| Multinomial Naive Bayes | 0.8858      | 0.8863    |  0.8858      | 0.8859        |
| Deep Neural Network | 0.9318      | 0.9320    |  0.9318     |   0.9318      |
| CNN | 0.9061    | 0.9067    |   0.9061    |     0.9060    |
| Bi-LSTM | 0.9276        | 0.9277   |  0.9275      |    0.9275     |
| CNN-BiLSTM Hybrid | 0.9061     | 0.9067      |  0.9061     |    0.9060     |


In this project, i have found **`93%`** accuracy with Deep Neural Network model which is the best score.
Report of classification and the Confusion Matrix shows that the models miss classified the category of  **`economy, state and education `**. It is because the text in this categories are very similar and contain similar words.
**In conclusion, I have achieved good accuracy with different models. This accuray can be further improved by doing hyperparameter tunning and by employing more shophisticated network architecture like Transformers and Pretrained Embeddings.**


