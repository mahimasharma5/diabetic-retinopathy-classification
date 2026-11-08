# diabetic-retinopathy-classification
ML classification and evaluation using retinal image features from UCI diabetic retinopathy dataset.

## 1. Problem Statement

Diabetic Retinopathy is an eye condition associated with diabetes that can lead to vision impairment if not detected early. This project focuses on applying machine learning classification techniques to identify signs of diabetic retinopathy using features extracted from retinal images.

The objective is to develop and evaluate five classification models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes and Support Vector Machine (SVM) using the Diabetic Retinopathy Debrecen dataset.

The performance of these models is compared using six evaluation metrics: Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). A Streamlit application is also developed to provide an interactive interface for evaluating and comparing the trained models on test data.

## 2. Dataset Description

The project uses the **Diabetic Retinopathy Debrecen dataset** from the UCI Machine Learning Repository. The dataset contains features extracted from retinal images in the Messidor image set and is designed for binary classification.

- **Original dataset:** 1,151 records
- **Records after removing exact duplicates:** 1,146
- **Number of features:** 19
- **Target variable:** `Class`
- **Classification type:** Binary classification
- **Class 0:** 540 records
- **Class 1:** 606 records

The 19 input features represent image quality assessment, pre-screening, microaneurysm detection, exudate detection, macula-optic disc distance, optic disc diameter and AM/FM-based classification.

The dataset was cleaned by removing five exact duplicate records. The cleaned data was then divided into training and testing sets using an 80:20 split, with stratification and a random state of 42. This resulted in 916 training records and 230 testing records.

## 3. GitHub Repository

**Repository:** https://github.com/mahimasharma5/diabetic-retinopathy-classification

The repository contains the exploratory data analysis notebook, model training notebook, trained model files, preprocessing scaler, test dataset, Streamlit application and project dependencies.

## 4. Models Used

5 ML classification models were trained and evaluated on the Diabetic Retinopathy Debrecen dataset. Their performance was compared using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7391 | 0.8153 | 0.8690 | 0.5984 | 0.7087 | 0.5146 |
| Decision Tree | 0.6000 | 0.6012 | 0.6339 | 0.5820 | 0.6068 | 0.2020 |
| KNN | 0.6043 | 0.6526 | 0.6566 | 0.5328 | 0.5882 | 0.2197 |
| Naive Bayes | 0.5696 | 0.6986 | 0.7805 | 0.2623 | 0.3926 | 0.2334 |
| Support Vector Machine | 0.6696 | 0.7807 | 0.7674 | 0.5410 | 0.6346 | 0.3670 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the best overall performance, with the highest Accuracy, AUC, Precision, Recall, F1-score and MCC among the five models. It provided the strongest overall classification results on this dataset. |
| Decision Tree | Achieved moderate performance. Its Recall was the second highest, but its Accuracy, AUC, Precision, F1-score and MCC were lower than those of Logistic Regression and Support Vector Machine. |
| KNN | Showed moderate performance, with results slightly better than Decision Tree in Accuracy, AUC, Precision and MCC but lower Recall and F1-score. |
| Naive Bayes | Achieved relatively high Precision but the lowest Recall and F1-score. This indicates that it identified fewer positive cases, despite having a comparatively high proportion of correct positive predictions. |
| Support Vector Machine | Achieved the second-highest AUC and the second-best overall results across several metrics. It performed better than Decision Tree, KNN and Naive Bayes on most evaluation metrics. |
| Overall, Winner for your dataset? | **Logistic Regression** was the overall winner for this dataset. It achieved the highest scores across all six evaluation metrics: Accuracy (0.7391), AUC (0.8153), Precision (0.8690), Recall (0.5984), F1-score (0.7087) and MCC (0.5146). |

These results indicate that Logistic Regression provided the strongest overall performance among the five evaluated models on the test dataset.
