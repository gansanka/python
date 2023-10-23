import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    "P1": [7, 7, 3, 1],
    "P2": [7, 4, 4, 4],
    "Class": ["False", "False", "True", "True"]
}

df = pd.DataFrame(data)

x = df.iloc[:,:-1].values
y = df.iloc[:,2].values

x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

print(knn.predict([[3,7]]))

