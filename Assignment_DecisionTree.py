import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #statstical visualization

headernames = ['looks', 'eloquence', 'alcohol-consumption', 'money-spent', 'date']

data = pd.read_csv('date.csv', names=headernames)
print(data.shape)
print(data.head())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data['looks'] = le.fit_transform(data['looks'])
data['eloquence'] = le.fit_transform(data['eloquence'])
data['alcohol-consumption'] = le.fit_transform(data['alcohol-consumption'])
data['date'] = le.fit_transform(data['date'])


X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

result1 = classification_report(y_test,y_pred)
print('Classification Report:',)
print(result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)