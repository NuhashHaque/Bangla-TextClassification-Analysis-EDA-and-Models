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
All the models parameters are tuned and optimized

## Model Evaluation 

| Model Name  | Accuracy    | Precision     | Recall | F1-Score|
| :---        |    :----:   |   ---:        |  ---:  |  ---:   |
| Header      | Title       | Here's this   |        |         |
| Paragraph   | Text        | And more      |        |         |

In this simple model we have got **`81%`** validation accuracy which is not bad for such an multiclass imbalanced dataset. Besides Confusion Matrix and other evaluation measures have been taken to determine the effectiveness of the developed model. From the confusion matrix it is observed that the maximum number of misclassified headlines are fall in the caltegory of **`Natinal, International and Politics `** and it makes sense because this categories headlines are kind of similar in words. The accuracy, precision, recall and f1-score result also demonstrate this issue. 

![confusion](/images/confusion.PNG)

![performance](/images/performance.PNG)

**In conclusion, we have achieved a good accuracy of `84%` on this simple recurrent neural network for Bengali news headline categorization task. This accuray can be further improved by doing hyperparameter tunning and by employing more shophisticated network architecture with a large dataset.**


