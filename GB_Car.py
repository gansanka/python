import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    'Color': ['Red','Red','Red','Yellow','Yellow','Yellow','Yellow','Yellow','Red','Red'],
    'Type': ['Sports','Sports','Sports','Sports','Sports','SUV','SUV','SUV','SUV','SUV'],
    'Origin': ['Domestic', 'Domestic', 'Domestic', 'Domestic','Imported', 'Imported','Imported','Domestic','Imported','Imported'],
    'Stolen': ['Yes', 'No', 'Yes','No', 'Yes','No','Yes','No','No','Yes']
}

#headernames = ['Color', 'Type', 'Origin', 'Stolen']
df = pd.DataFrame(data)
#df = pd.read_csv('cars.csv', names=headernames, dtype=str)

print(df)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['Color'] = le.fit_transform(df['Color'])
df['Type'] = le.fit_transform(df['Type'])
df['Origin'] = le.fit_transform(df['Origin'])
df['Stolen'] = le.fit_transform(df['Stolen'])

x = df.iloc[:,:-1].values
y = df.iloc[:,3].values
x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

check = le.fit_transform(['Red','SUV','Domestic'])
y_pred = classifier.predict([check])
print(y_pred)
