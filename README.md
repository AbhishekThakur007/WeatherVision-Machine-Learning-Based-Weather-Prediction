# WeatherVision: Machine Learning-Based Weather Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Tools and Libraries](#tools-and-libraries)
4. [Project Workflow](#project-workflow)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)

---

## 1. Introduction
WeatherVision showcases how machine learning models can be utilized to predict weather conditions accurately. This project is designed for beginners and enthusiasts in data science and machine learning, aiming to provide a step-by-step approach to weather prediction using various machine learning techniques.

### What is Weather Prediction?
Weather prediction, also known as weather forecasting, involves predicting the atmospheric conditions at a specific location and time in the future. Accurate weather prediction is crucial for applications like:
- **Agriculture**: Assisting in irrigation and harvest planning.
- **Transportation**: Scheduling and route optimization.
- **Aviation**: Ensuring the safety and efficiency of air travel.
- **Disaster Management**: Preparing for extreme weather events such as hurricanes, floods, and storms.

Machine learning excels in weather prediction by handling complex relationships in large datasets sourced from various sources like satellites, radars, and weather stations.

---

## 2. Dataset Description
The dataset used contains weather data for Seattle, comprising 1461 rows with the following attributes:
- **Date**: Specific date of observation.
- **Precipitation**: Amount of precipitation (rain, hail, etc.).
- **temp_max**: Maximum temperature.
- **temp_min**: Minimum temperature.
- **wind**: Wind speed.
- **weather**: Type of weather (drizzle, rain, sun, snow, fog).

You can download the dataset [here](#).

---

## 3. Tools and Libraries
The project utilizes the following Python libraries:
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **Plotly**: For interactive visualizations.
- **Scikit-Learn**: For machine learning models and preprocessing.

---

## 4. Project Workflow

### Step 1: Import Necessary Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
```

### Step 2: Load and Explore the Dataset
```python
# Load the dataset
data = pd.read_csv("weather.csv")
data.head()
data.info()

# Check for missing values
data.isnull().sum()
```

### Step 3: Data Preprocessing
```python
# Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Encode categorical variables
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['weather'] = label_encoder.fit_transform(data['weather'])

# Drop irrelevant columns
data = data.drop('date', axis=1)
```

### Step 4: Split the Dataset
```python
X = data.drop('weather', axis=1)
y = data['weather']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

### Step 5: Normalize the Data
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Step 6: Train and Evaluate Models
#### Logistic Regression
```python
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='.3g')
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### Naive Bayes
```python
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='.3g')
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### Support Vector Classifier
```python
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='.3g')
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 5. Conclusion
The WeatherVision project demonstrates the potential of machine learning in weather forecasting. The models achieved the following accuracies:
- **Logistic Regression** and **Support Vector Classifier** achieved 79.5% accuracy.
- **Naive Bayes** performed slightly better with an accuracy of 84.15%.

These results highlight the effectiveness of machine learning models in handling complex atmospheric data.

---

## 6. Future Work
- Experiment with advanced models like Random Forest or Gradient Boosting.
- Use larger, more diverse datasets.
- Explore deep learning techniques for time-series forecasting.