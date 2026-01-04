# Suicide and Depression Text Classification using Machine Learning

## Overview
This project applies machine learning and natural language processing techniques
to automatically detect suicidal ideation in Reddit posts. The goal is to compare
classical text representations (BoW, TF-IDF) with contextual embeddings (BERT)
across multiple linear classifiers.

## Dataset
- Source: Kaggle (Reddit SuicideWatch dataset)
- Size: 232,074 posts
- Classes: Suicidal vs Non-Suicidal (balanced)

## Methods
- Text preprocessing and cleaning
- Feature extraction:
  - Bag-of-Words
  - TF-IDF
  - BERT embeddings
- Models:
  - Logistic Regression
  - Linear Support Vector Classifier (LinearSVC)

## Results
- Best model: **TF-IDF + LinearSVC**
- Accuracy: **93.7%**
- Lowest test error among all models
- BERT underperformed classical methods due to computational constraints

## Tools & Technologies
- Python
- scikit-learn
- pandas, NumPy
- HuggingFace Transformers
- Matplotlib / Seaborn

## Ethical Considerations
This project is for research purposes only and does not replace professional
mental health diagnosis or intervention.

## Future Work
- Explore neural network classifiers
- Expand to other social media platforms
- Real-time moderation or alerting systems
