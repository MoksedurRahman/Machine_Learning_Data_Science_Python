# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

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
get_ipython().run_line_magic('matplotlib', 'inline')


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
fig = plt.figure(1, figsize=(20,8))
plt.scatter(X[['SL']], encodedY)
plt.xlabel("SepalLength")
plt.ylabel("Species")
plt.show()


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
# Spliting data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    X[["PL", "PW"]], encodedY, test_size=0.2, random_state=42)


# %%
# MOdeling
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)


# %%
ypred == ytest


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
ypred


# %%
encoder.inverse_transform(ypred)


# %%



