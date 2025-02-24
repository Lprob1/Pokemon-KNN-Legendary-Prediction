from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#plotting
import matplotlib.pyplot as plt

df = pd.read_csv("Pokemon.csv")

print(df.head())
print(df.columns)
#goal: predict legend based on the rest
#get all the values and put in list and array form
y = df["Legendary"].tolist()
X = df[["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]].to_numpy()

#create a map of Type 1 to number
number_to_type = {i:target for i, target in enumerate(np.unique(y))}
type_to_number = {number_to_type[key]:key for key in number_to_type}

#change all names to numbers in y
y = [type_to_number[t] for t in y]

# #turn Type 1 and Type 2 into numbers
# types = list(set(np.unique(df["Type 1"])).union(set(np.unique(df["Type 2"])))) #get all the unique types


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#create a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


#now, knn
min_neighbors = 3
max_neighbors = 7
#for PCA
min_components = 1
max_components = 5
best_pca = None

best_classifier = None
best_accuracy = 0
best_n = 0
best_n_components = 0


n_components = min_components
while n_components <= max_components:
    n_neighbors = min_neighbors
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_train)
    while n_neighbors <= max_neighbors:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors) 
        classifier.fit(X_pca, y_train)
        X_val_pca = pca.transform(X_val)
        y_pred = classifier.predict(X_val_pca)
        acc = accuracy_score(y_true = y_val, y_pred=y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_classifier = classifier
            best_pca = pca
            best_n = n_neighbors
            best_n_components = n_components
        n_neighbors += 1
    n_components += 1

print(f"The optimal accuracy of {best_accuracy} is achieved on validation set with {best_n} neighbors and {best_n_components} principal components")
X_test = best_pca.transform(X_test)
y_pred = best_classifier.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"The final accuracy is {test_acc}")