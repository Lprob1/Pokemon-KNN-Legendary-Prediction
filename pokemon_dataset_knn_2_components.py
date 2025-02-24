from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

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

#train a PCA on X_train, transform
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

#plot the train set to see if there are some regularities
fig, ax = plt.subplots()
#get a color palette of distinct colors
palette = sns.color_palette("husl", len(np.unique(y)))

for i, target in enumerate(np.unique(y)):
    ax.scatter(X_pca[y_train==target, 0], X_pca[y_train==target, 1],
               label=number_to_type[target], color=palette[i], alpha=0.7)
    
plt.legend()
plt.show()

#now, knn
min_neighbors = 3
max_neighbors = 7

best_classifier = None
best_accuracy = 0
best_n = 0
curr_neighbors = min_neighbors

X_val = pca.transform(X_val)

while curr_neighbors <= max_neighbors:
    classifier = KNeighborsClassifier(n_neighbors=curr_neighbors)
    classifier.fit(X_pca, y_train)
    y_pred = classifier.predict(X_val)
    acc = accuracy_score(y_true = y_val, y_pred=y_pred)
    if acc > best_accuracy:
        best_accuracy = acc
        best_classifier = classifier
        best_n = curr_neighbors
    curr_neighbors += 1

print(f"The optimal accuracy of {best_accuracy} is achieved on validation set with {best_n} neighbors.")
X_test = pca.transform(X_test)
y_pred = best_classifier.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"The final accuracy is {test_acc}")

#plot the decision boundary to see what it looks like
n_components = 2
plot_decision_regions(X_pca, np.asarray(y_train), clf=best_classifier, legend=n_components)
plt.show()