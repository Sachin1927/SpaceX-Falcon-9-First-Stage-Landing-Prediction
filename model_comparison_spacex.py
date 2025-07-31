
#  SpaceX Launch Classification - Model Comparison


#  Install packages (for JupyterLite if needed)
import piplite
await piplite.install(['numpy', 'pandas', 'seaborn'])

#  Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from js import fetch
import io


#  Load the datasets


# Dataset with target variable
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = await fetch(URL1)
data = pd.read_csv(io.BytesIO((await resp1.arrayBuffer()).to_py()))

# Dataset with feature variables
URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = await fetch(URL2)
X = pd.read_csv(io.BytesIO((await resp2.arrayBuffer()).to_py()))

Y = data['Class'].to_numpy()


#  Preprocess: Standardize and Split


# Standardize features
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# Utility: Confusion Matrix Plot


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Did Not Land', 'Landed'])
    ax.yaxis.set_ticklabels(['Did Not Land', 'Landed'])
    plt.show()


#  Logistic Regression


lr_params = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, lr_params, cv=10)
logreg_cv.fit(X_train, Y_train)

print(" Logistic Regression Best Params:", logreg_cv.best_params_)
print("Training Accuracy:", logreg_cv.best_score_)
print("Test Accuracy:", logreg_cv.score(X_test, Y_test))
plot_confusion_matrix(Y_test, logreg_cv.predict(X_test))


#  Support Vector Machine


svm_params = {
    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    'C': np.logspace(-3, 3, 5),
    'gamma': np.logspace(-3, 3, 5)
}
svm = SVC()
svm_cv = GridSearchCV(svm, svm_params, cv=10)
svm_cv.fit(X_train, Y_train)

print(" SVM Best Params:", svm_cv.best_params_)
print("Training Accuracy:", svm_cv.best_score_)
print("Test Accuracy:", svm_cv.score(X_test, Y_test))
plot_confusion_matrix(Y_test, svm_cv.predict(X_test))


#  Decision Tree


tree_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2 * i for i in range(1, 10)],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, tree_params, cv=10)
tree_cv.fit(X_train, Y_train)

print("🟢 Decision Tree Best Params:", tree_cv.best_params_)
print("Training Accuracy:", tree_cv.best_score_)
print("Test Accuracy:", tree_cv.score(X_test, Y_test))
plot_confusion_matrix(Y_test, tree_cv.predict(X_test))


#  K-Nearest Neighbors


knn_params = {
    'n_neighbors': list(range(1, 11)),
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, Y_train)

print(" KNN Best Params:", knn_cv.best_params_)
print("Training Accuracy:", knn_cv.best_score_)
print("Test Accuracy:", knn_cv.score(X_test, Y_test))
plot_confusion_matrix(Y_test, knn_cv.predict(X_test))


# Final Comparison


results = {
    "Logistic Regression": logreg_cv.score(X_test, Y_test),
    "Support Vector Machine": svm_cv.score(X_test, Y_test),
    "Decision Tree": tree_cv.score(X_test, Y_test),
    "K-Nearest Neighbors": knn_cv.score(X_test, Y_test)
}

print("\n Final Model Comparison:")
for model, acc in results.items():
    print(f"{model} Test Accuracy: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\n Best Performing Model: {best_model} with Accuracy = {results[best_model]:.4f}")
