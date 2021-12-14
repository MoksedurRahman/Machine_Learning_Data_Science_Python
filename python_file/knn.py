# %% [markdown]
# # 8 Steps Of Machine Learning
# * Data Gathering
# * Data pre-processing
# * Feature Engineering
# * Choosing Model
# * Training Model
# * Test Model/ Model Evaluation
# * Parameter Tuning
# * Prediction

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%matplotlib inline

# %%
# Loading Data
data = pd.read_csv(r'.\data\iris.csv')
data.head()

# %%
data.drop(columns=['Id'], inplace=True)
data.head()

# %%
data.columns = ['SL', 'SW', 'PL', 'PW', 'SP']

# %%
data.info()

# %%
data[['SP']].value_counts()

# %%
# Data preprocessing
#data[['Species']] = data[['Species']].replace({"Iris-setosa":0,"Iris-versicolor":1, "Iris-virginica":2})

# %%
X = data[data.columns[:4]]
X

# %%
Y = data[[data.columns[-1]]]
Y

# %%
# Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encodedY = encoder.fit_transform(Y.values.ravel())
encodedY

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].scatter(data[data["SP"] == "Iris-setosa"][['SL']],
                data[data["SP"] == "Iris-setosa"][['SW']], color='cyan', label="Iris-setosa")
axes[0].scatter(data[data["SP"] == "Iris-versicolor"][['SL']],
                data[data["SP"] == "Iris-versicolor"][['SW']], color='darkorange', label="Iris-versicolor")
axes[0].scatter(data[data["SP"] == "Iris-virginica"][['SL']],
                data[data["SP"] == "Iris-virginica"][['SW']], color='brown', label="Iris-virginica")
axes[0].set_xlabel("SepalLength")
axes[0].set_ylabel("SepalWidth")

axes[1].scatter(data[data["SP"] == "Iris-setosa"][['PL']],
                data[data["SP"] == "Iris-setosa"][['PW']], color='cyan', label="Iris-setosa")
axes[1].scatter(data[data["SP"] == "Iris-versicolor"][['PL']],
                data[data["SP"] == "Iris-versicolor"][['PW']], color='darkorange', label="Iris-versicolor")
axes[1].scatter(data[data["SP"] == "Iris-virginica"][['PL']],
                data[data["SP"] == "Iris-virginica"][['PW']], color='brown', label="Iris-virginica")
axes[1].set_xlabel("PetalLength")
axes[1].set_ylabel("PetalWidth")
plt.show()
plt.show()


# %%
# Standard Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaledFeatures = scaler.fit_transform(X)
scaledFeatures

# %%
# Spliting data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    X[["PL", "PW"]], encodedY, test_size=0.2, random_state=42)


# %%
# MOdeling
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(xtrain, ytrain)

# %%
ypred = model.predict(xtest)
ypred == ytest

# %%
model.predict_proba(xtest)[1].argmax()

# %%
#Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(ytest, ypred)

# %%
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, xtest, ytest)

# %%
accuracy_score(ytest, ypred)

# %%
model.get_params()

# %%
# Parameter optimization
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    model,
    param_grid={
        'n_neighbors': [3,5,7,9,11,13,15,20],
        'weights': ['uniform', 'distance']
    }, cv=6, scoring='accuracy'
)
grid.fit(xtrain, ytrain)

# %%
grid.best_params_

# %%
grid.best_score_

# %%
grid.best_estimator_

# %%
encoder.inverse_transform(ypred)

# %%
# Cross Validaidation
from sklearn.model_selection import cross_val_score

cv = cross_val_score(model, X[["PL", "PW"]], encodedY, cv=6)
cv, cv.mean()

# %%
# Visualization
bmodel = grid.best_estimator_
bmodel.fit(xtrain, ytrain.ravel())
xmin, xmax = X[['PL']].values.min(), X[['PL']].values.max()
ymin, ymax = X[['PW']].values.min(), X[['PW']].values.max()

xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
xx.shape, yy.shape

xtest = np.c_[xx.ravel(), yy.ravel()]
xtest.shape
ypred = model.predict(xtest)
ypred = ypred.reshape(yy.shape)
ypred

plt.figure(1, figsize=(20,8))
plt.set_cmap(plt.cm.Accent_r)
plt.pcolormesh(xx, yy, ypred, shading='auto')

plt.scatter(X[['PL']], X[['PW']], c=encodedY, edgecolors='black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# %%
dir(plt.cm)

# %%



