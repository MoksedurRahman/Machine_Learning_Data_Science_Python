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
# Data Loading
data = pd.read_csv(r'.\data\FuelConsumption.csv')
data.head()


# %%
data.info()


# %%
data.describe()


# %%
data.drop(columns=["MODELYEAR","MAKE","MODEL", "TRANSMISSION"], inplace=True)


# %%
data[['FUELTYPE']].value_counts()


# %%
data[['VEHICLECLASS']].value_counts()


# %%
data.drop(columns=['VEHICLECLASS', 'FUELTYPE'], inplace=True)


# %%
data.head()


# %%
data.describe()


# %%
data[['ENGINESIZE']].hist()


# %%
data[['CYLINDERS']].hist()


# %%
data[['FUELCONSUMPTION_COMB']].hist()


# %%
# Feature Selection
data.corr()


# %%
sns.heatmap(data.corr(), annot=True)


# %%
data.drop(columns=['FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG'], inplace=True)


# %%
sns.heatmap(data.corr(), annot=True)


# %%
data.head()


# %%
feature = data.drop(columns=['CO2EMISSIONS'])
feature


# %%
target = data[['CO2EMISSIONS']]
target


# %%
# Spliting Data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(feature, target, test_size=0.25, random_state=1)


# %%
xtrain.shape, xtest.shape


# %%
ytrain.shape, ytest.shape


# %%
# Choosing Model

plt.scatter(feature.ENGINESIZE ,target.values)
plt.title("EngineSize vs Co2Emission")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSION")
plt.show()


# %%
# Modeling
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain[['ENGINESIZE']], ytrain.values.ravel())


# %%
model.intercept_


# %%
model.coef_


# %%
# Prediction
ypred = model.predict(xtest[['ENGINESIZE']])
xtest[['ENGINESIZE']].iloc[0], ypred[0], ytest.values[0]


# %%
ytest.values[0] - ypred[0] 


# %%
# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
print("R2 Score: ", r2_score(ytest, ypred))


# %%
model2 = LinearRegression()
model2.fit(xtrain[['CYLINDERS']], ytrain.values.ravel())
# Prediction
ypred = model2.predict(xtest[['CYLINDERS']])

print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
print("R2 Score: ", r2_score(ytest, ypred))


# %%
# Linear Regression with FUELCONSUMPTION_CITY


# %%
xtrain


# %%
# Multiple Linear Regression
model4 = LinearRegression()
model4.fit(xtrain, ytrain.values.ravel())


# %%
model4.intercept_


# %%
model4.coef_


# %%
# Prediction
ypred = model4.predict(xtest)

print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
print("R2 Score: ", r2_score(ytest, ypred))


# %%
# Cross Validation
from sklearn.model_selection import KFold

accuracies = []
folds = KFold(n_splits=5)
for train_index, test_index in folds.split(feature):
    # Extracting Training Data
    xtrain = feature.iloc[train_index]
    ytrain = target.iloc[train_index]
    #
    xtest = feature.iloc[test_index]
    ytest = target.iloc[test_index]
    print()
    # MOdeling
    model5 = LinearRegression()
    model5.fit(xtrain, ytrain)

    # Evaluation
    print('=====================================')
    ypred = model5.predict(xtest)
    print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
    print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
    print("R2 Score: ", r2_score(ytest, ypred))
    accuracies.append(r2_score(ytest, ypred))


# %%
print("Average R2 Score: ", sum(accuracies)/5)


# %%
# Cross VAlidation
from sklearn.model_selection import cross_val_score
cvmodel = LinearRegression()
cv = cross_val_score(cvmodel, feature, target, cv=10)
cv, cv.mean()


# %%
# Model Visualization
model6 = LinearRegression()
model6.fit(feature[['ENGINESIZE']], target)


# %%
fig = plt.figure(1, figsize=(20,8))
plt.scatter(feature[['ENGINESIZE']].values, target.values, label='Actual Data', color='darkorange')
plt.scatter(feature[['ENGINESIZE']].values, model6.predict(feature[['ENGINESIZE']]), label='Predicted Data', color='black')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSION")
plt.legend()
plt.show()


# %%
dummyFeature = np.linspace(0, 10, 400).reshape(-1,1)
dummyFeature


# %%
fig = plt.figure(1, figsize=(20,8))
plt.scatter(feature[['ENGINESIZE']].values, target.values, label='Actual Data', color='darkorange')
plt.plot(dummyFeature, model6.predict(dummyFeature), label='Predicted Data', color='black')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSION")
plt.legend()
plt.show()


# %%
dummyFeature1 = np.linspace(0, 10, 400)
dummyFeature2 = np.linspace(4, 31, 400)
dummyFeature = np.c_[dummyFeature1, dummyFeature2]


# %%
model7 = LinearRegression().fit(
    feature[['ENGINESIZE', 'FUELCONSUMPTION_CITY']], target)
fig = plt.figure(1, figsize=(20, 5))
axes = plt.axes(projection='3d')
axes.scatter3D(feature[['ENGINESIZE']], feature[[
               'FUELCONSUMPTION_CITY']], target, label='Actual Data', color='darkorange')
axes.plot3D(dummyFeature1, dummyFeature2, model7.predict(
    dummyFeature).ravel(), label='Prediction', color='black')
axes.legend()
axes.set_xlabel("ENGINESIZE")
axes.set_ylabel("FUELCONSUMMPTION_CITY")
axes.set_zlabel("CO2EMISSION")
plt.show()


# %%
# Saving MOdel
from joblib import dump
dump(model5, 'Lr')


# %%
from joblib import load
model = load('Lr')


# %%
model.predict(xtest)


# %%



