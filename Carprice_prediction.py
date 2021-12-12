# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# 
# %% [markdown]
# #   Car Price Prediction
# 
# ### Problem Statement
# 
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
# 
# - Which variables are significant in predicting the price of a car
# - How well those variables describe the price of a car
# 
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market.
# 
# ### Business Goal
# 
# You are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

# %%
import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ### Step 1: Reading and Understanding the Data
# 
# Let's start with the following steps:
# 
# 1. Importing data using the pandas library
# 2. Understanding the structure of the data

# %%
cars = pd.read_csv('./cardata/CarPrice_Assignment.csv')
cars.head(10)


# %%
cars.shape


# %%
cars.describe()


# %%
cars.info()

# %% [markdown]
# ### Step 2 : Data Cleaning and Preparation

# %%
# Spittiling Company Name from CarName Column
CompanyName=cars['CarName'].apply(lambda x: x.split(' ')[0])
cars.insert(3,'CompanyName',CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)
cars.head()


# %%
cars.CompanyName.unique()

# %% [markdown]
#     Fixing invalid values
# 
# - There seems to be some spelling error in the CompanyName column.
# 
#     - maxda = mazda
#     - Nissan = nissan
#     - porsche = porcshce
#     - toyota = toyouta
#     - vokswagen = volkswagen = vw

# %%
cars.CompanyName=cars.CompanyName.str.lower()

def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)
    
replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

cars.CompanyName.unique()


# %%
cars.loc[cars.duplicated()]


# %%
cars.columns


# %%
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(cars.price)

plt.subplot(1,2,2)
plt.title('Cars Price Spread')
sns.boxplot(y=cars.price)

plt.show()


# %%
cars.price.describe(percentiles=[0.25,0.50,0.75,0.85,0.90,1])

# %% [markdown]
# #### Inference :
#     
# 1.  The plot seemed to be right-skewed, meaning that the most prices in the dataset are low(Below 15,000).
# 
# 2.  There is a significant difference between the mean and the median of the price distribution.
# 
# 3.  The data points are far spread out from the mean, which indicates a high variance in the car prices.(85% of the prices are below 18,500, whereas the remaining 15% are between 18,500 and 45,400.)
# %% [markdown]
# #### Step 3.1 : Visualising Categorical Data
# 
# - CompanyName
# - Symboling
# - fueltype
# - enginetype
# - carbody
# - doornumber
# - enginelocation
# - fuelsystem
# - cylindernumber
# - aspiration
# - drivewheel

# %%
plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = cars.CompanyName.value_counts().plot(kind='bar')
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car company', ylabel='Frequency of company')

plt.subplot(1,3,2)
plt1 = cars.fueltype.value_counts().plot(kind='bar')
plt.title('Fuel Type Histogram')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')

plt.subplot(1,3,3)
plt1 = cars.carbody.value_counts().plot(kind='bar')
plt.title('Car Type Histogram')
plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')

plt.show()

# %% [markdown]
# #### Inference :
# 1.  Toyota seemed to be favored car company.
# 2.  Number of gas fueled cars are more than diesel.
# 3.  sedan is the top car type prefered.

# %%
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Symboling Histogram')
sns.countplot(cars.symboling, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('Symboling vs Price')
sns.boxplot(x=cars.symboling, y=cars.price, palette=("cubehelix"))

plt.show()

# %% [markdown]
# ####    Inference :
# 1.  It seems that the symboling with 0 and 1 values have high number of rows (i.e. They are most sold.)
# 2.  The cars with -1 symboling seems to be high priced (as it makes sense too,
#  insurance risk rating -1 is quite good). But it seems that symboling with 3 value has the price range similar to -2 value. There is a dip in price at symboling 1

# %%
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Engine Type Histogram')
sns.countplot(cars.enginetype, palette=("Blues_d"))

plt.subplot(1,2,2)
plt.title('Engine Type vs Price')
sns.boxplot(x=cars.enginetype, y=cars.price, palette=("PuBuGn"))

plt.show()

df = pd.DataFrame(cars.groupby(['enginetype'])['price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Engine Type vs Average Price')
plt.show()

# %% [markdown]
# ### Inference :
# 1. ohc Engine type seems to be most favored type.
# 2. ohcv has the highest price range (While dohcv has only one row), ohc and ohcf have the low price range.

# %%
plt.figure(figsize=(25, 6))

df = pd.DataFrame(cars.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()

df = pd.DataFrame(cars.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Fuel Type vs Average Price')
plt.show()

df = pd.DataFrame(cars.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Car Type vs Average Price')
plt.show()


