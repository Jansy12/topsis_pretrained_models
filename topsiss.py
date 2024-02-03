#!/usr/bin/env python
# coding: utf-8

# In[25]:


from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset


# In[27]:


ag_news_dataset = load_dataset("ag_news")

# Extract text and labels
texts = ag_news_dataset['train']['text'][:2000]  # Sample only the first 1000 samples
labels = ag_news_dataset['train']['label'][:2000]
train_df = pd.DataFrame(ag_news_dataset["train"])

# Print the first few rows of the DataFrame
print("First few rows of the training split:")
print(train_df.head())

# Plot a bar chart to visualize class distribution in the training split
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_df)
plt.title('Class Distribution in Training Split')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
from datasets import load_dataset

# Load the AG News dataset
ag_news_dataset = load_dataset("ag_news")

# Print the entire training split
print("Training Split:")
for example in ag_news_dataset["train"]:
    print(example)

# Print the entire testing split
print("Testing Split:")
for example in ag_news_dataset["test"]:
    print(example)


# In[8]:


classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier()
}

# Initialize weights for performance metrics
weights = {
    'Accuracy': 0.3,
    'Precision': 0.3,
    'Recall': 0.2,
    'F1-score': 0.2
}


# In[9]:


accuracy_list = []
precision_list = []
recall_list = []
f1_list = []


# In[12]:


for name, classifier in classifiers.items():
    # Train the model
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, average='weighted'))
    recall_list.append(recall_score(y_test, y_pred, average='weighted'))
    f1_list.append(f1_score(y_test, y_pred, average='weighted'))


# In[13]:


max_accuracy = max(accuracy_list)
min_accuracy = min(accuracy_list)
max_precision = max(precision_list)
min_precision = min(precision_list)
max_recall = max(recall_list)
min_recall = min(recall_list)
max_f1 = max(f1_list)
min_f1 = min(f1_list)


# In[14]:


normalized_metrics = {
    'Accuracy': [(x - min_accuracy) / (max_accuracy - min_accuracy) for x in accuracy_list],
    'Precision': [(x - min_precision) / (max_precision - min_precision) for x in precision_list],
    'Recall': [(x - min_recall) / (max_recall - min_recall) for x in recall_list],
    'F1-score': [(x - min_f1) / (max_f1 - min_f1) for x in f1_list]
}


# In[15]:


weighted_scores = {}
for name in classifiers.keys():
    weighted_score = 0
    for metric, weight in weights.items():
        weighted_score += normalized_metrics[metric][list(classifiers.keys()).index(name)] * weight
    weighted_scores[name] = weighted_score


# In[16]:


print("TOPSIS Results:")
for name, score in sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score}")


# In[17]:


ideal_solution = {}
negative_ideal_solution = {}

for metric in normalized_metrics.keys():
    ideal_solution[metric] = max(normalized_metrics[metric])
    negative_ideal_solution[metric] = min(normalized_metrics[metric])

# Calculate the distance from the ideal and negative ideal solutions for each model
ideal_distances = {}
negative_ideal_distances = {}

for name in classifiers.keys():
    ideal_distances[name] = np.sqrt(sum((normalized_metrics[metric][list(classifiers.keys()).index(name)] - ideal_solution[metric]) ** 2 for metric in weights.keys()))
    negative_ideal_distances[name] = np.sqrt(sum((normalized_metrics[metric][list(classifiers.keys()).index(name)] - negative_ideal_solution[metric]) ** 2 for metric in weights.keys()))


# In[18]:


topsis_scores = {}

for name in classifiers.keys():
    topsis_scores[name] = negative_ideal_distances[name] / (ideal_distances[name] + negative_ideal_distances[name])

# Rank the models based on their TOPSIS score
ranked_models_topsis = sorted(topsis_scores.items(), key=lambda x: x[1], reverse=True)


# In[19]:


# Print the TOPSIS rank
print("\nTOPSIS Rank:")
for i, (name, score) in enumerate(ranked_models_topsis, start=1):
    print(f"{i}. {name}: {score}")


# In[ ]:




