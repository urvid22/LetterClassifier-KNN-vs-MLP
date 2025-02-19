#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
url = r"C:\Users\urvi9\letters.csv"
df = pd.read_csv(url)
df


# In[2]:


X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)


# In[3]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300)
mlp.fit(X_train_scaled, y_train)
mlp_predictions = mlp.predict(X_test_scaled)

# Calculate accuracy for Neural Network model
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print("Neural Network Accuracy:", mlp_accuracy)


# In[4]:


print("\nClassification Report for KNN:")
print(classification_report(y_test, knn_predictions))

print("\nClassification Report for Neural Network:")
print(classification_report(y_test, mlp_predictions))

# Confusion Matrix for KNN
print("\nConfusion Matrix for KNN:")
print(confusion_matrix(y_test, knn_predictions))

# Confusion Matrix for Neural Network
print("\nConfusion Matrix for Neural Network:")
print(confusion_matrix(y_test, mlp_predictions))


# In[ ]:




