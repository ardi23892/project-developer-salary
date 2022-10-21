#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("survey_results_public.csv")


# In[2]:


df.head()


# In[3]:


df = df[["Country","EdLevel","YearsCodePro","Employment","ConvertedCompYearly"]]
df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
df.head()


# In[4]:


df = df[df["Salary"].notnull()]
df.head()


# In[5]:


df.info()


# In[6]:


df = df.dropna()
df.isnull().sum()


# In[7]:


df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()


# In[8]:


df["Country"].value_counts()


# In[9]:


def CutCountry(categories, minimum):
    NewCountry = {}
    for i in range(len(categories)):
        if categories.values[i] >= minimum:
            NewCountry[categories.index[i]] = categories.index[i]
        else:
            NewCountry[categories.index[i]] = 'Other'
    return NewCountry


# In[10]:


CuttedCountry = CutCountry(df.Country.value_counts(),500)
df['Country'] = df['Country'].map(CuttedCountry)
df.Country.value_counts()


# In[11]:


fig, ax = plt.subplots(1,1,figsize=(12,7))
df.boxplot("Salary", "Country", ax=ax)
plt.suptitle("Salary ( US $) & Country")
plt.title("")
plt.ylabel("Salary")
plt.xticks(rotation = 90)
plt.show()


# In[12]:


df = df[df["Salary"] <= 300000]
df = df[df["Salary"] >= 10000]
df = df[df["Country"] != "Other"]


# In[13]:


fig, ax = plt.subplots(1,1,figsize=(12,7))
df.boxplot("Salary", "Country", ax=ax)
plt.suptitle("Salary ( US $) & Country")
plt.title("")
plt.ylabel("Salary")
plt.xticks(rotation = 90)
plt.show()


# In[14]:


df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]


# In[15]:


#Australia
Australia = df[ (df['Salary'] >= 146000) & (df["Country"] == "Australia") ].index
df.drop(Australia, inplace = True) 


# In[16]:


#Brazil
Brazil = df[ (df['Salary'] >= 58000) & (df["Country"] == "Brazil") ].index
df.drop(Brazil, inplace = True)


# In[17]:


#Canada
Canada = df[ (df['Salary'] >= 147000) & (df["Country"] == "Canada") ].index
df.drop(Canada, inplace = True) 


# In[18]:


#France
France = df[ (df['Salary'] >= 88000) & (df["Country"] == "France") ].index
df.drop(France, inplace = True) 


# In[19]:


#Germany
GermanyTop = df[ (df['Salary'] >= 116000) & (df["Country"] == "Germany") ].index
df.drop(GermanyTop, inplace = True)
GermanyBot = df[ (df['Salary'] <= 20000) & (df["Country"] == "Germany") ].index
df.drop(GermanyBot, inplace = True)


# In[20]:


#India
India = df[ (df['Salary'] >= 47000) & (df["Country"] == "India") ].index
df.drop(India, inplace = True)


# In[21]:


#Italy
Italy = df[ (df['Salary'] >= 70000) & (df["Country"] == "Italy") ].index
df.drop(Italy, inplace = True) 


# In[22]:


#Netherlands
Netherlands = df[ (df['Salary'] >= 120000) & (df["Country"] == "Netherlands") ].index
df.drop(Netherlands, inplace = True) 


# In[23]:


#Poland
Poland = df[ (df['Salary'] >= 80000) & (df["Country"] == "Poland") ].index
df.drop(Poland, inplace = True) 


# In[24]:


#Russian Federation
RussianFed = df[ (df['Salary'] >= 86000) & (df["Country"] == "Russian Federation") ].index
df.drop(RussianFed, inplace = True) 


# In[25]:


#Spain
Spain = df[ (df['Salary'] >= 84000) & (df["Country"] == "Spain") ].index
df.drop(Spain, inplace = True) 


# In[26]:


#Sweden
SwedenTop = df[ (df['Salary'] >= 90000) & (df["Country"] == "Sweden") ].index
df.drop(SwedenTop, inplace = True) 
SwedenBot = df[ (df['Salary'] <= 20000) & (df["Country"] == "Sweden") ].index
df.drop(SwedenBot, inplace = True) 


# In[27]:


#United Kingdom
UK = df[ (df['Salary'] >= 149000) & (df["Country"] == "United Kingdom of Great Britain and Northern Ireland") ].index
df.drop(UK, inplace = True)


# In[28]:


fig, ax = plt.subplots(1,1,figsize=(12,7))
df.boxplot("Salary", "Country", ax=ax)
plt.suptitle("Salary ( US $) & Country")
plt.title("")
plt.ylabel("Salary")
plt.xticks(rotation = 90)
plt.show()


# In[29]:


df["YearsCodePro"].unique()


# In[30]:


def StF(x):
    if x == "More than 50 years":
        return 51
    if x == "Less than 1 year":
        return 0.5
    return float(x)

df["YearsCodePro"] = df["YearsCodePro"].apply(StF)


# In[31]:


fig, ax = plt.subplots(1,1,figsize=(12,7))
df.boxplot("YearsCodePro")
plt.show()


# In[32]:


#Dropping outlier in Experience
df = df[df["YearsCodePro"] <= 25]


# In[33]:


fig, ax = plt.subplots(1,1,figsize=(12,7))
df.boxplot("YearsCodePro")
plt.show()


# In[34]:


df["EdLevel"].unique()


# In[35]:


def CombineEdLevel(x):
    if "Bachelor" in x:
        return "Bachelor's degree"
    if "Master" in x:
        return "Master's degree"
    if "Professional" in x or "Other doctoral" in x:
        return "Professional or Doctoral Degree"
    return "Less than a Bachelor"

df["EdLevel"] = df["EdLevel"].apply(CombineEdLevel)


# In[36]:


df["EdLevel"].unique()


# In[37]:


df.head()


# In[38]:


df.Country.unique()


# In[39]:


from sklearn.preprocessing import LabelEncoder
EdLevel_LE = LabelEncoder()
df["EdLevel"] = EdLevel_LE.fit_transform(df.EdLevel.values)
df["EdLevel"].unique()


# In[40]:


Country_LE = LabelEncoder()
df["Country"] = Country_LE.fit_transform(df.Country.values)
df["Country"].unique()


# In[41]:


x = df.drop("Salary", axis=1)
y = df["Salary"]


# In[42]:


x


# In[43]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[44]:


xTrain


# In[45]:


xTest


# In[46]:


yTrain


# In[47]:


yTest


# In[48]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(xTrain, yTrain.values)


# In[49]:


y_pred = LR.predict(xTest)
x_pred = LR.predict(xTrain)


# In[50]:


y_pred


# In[51]:


x_pred


# In[52]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
err = np.sqrt(mean_squared_error(yTest,y_pred))


# In[53]:


err


# In[54]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

MD = [None, 2, 4, 6, 8, 10]
parameters = {"max_depth": MD}

regressor = DecisionTreeRegressor(random_state=0)
GS = GridSearchCV(regressor, parameters, scoring="neg_mean_squared_error")
GS.fit(xTrain, yTrain.values)


# In[55]:


regressor = GS.best_estimator_
regressor.fit(xTrain, yTrain.values)
y_pred = regressor.predict(xTest)
err = np.sqrt(mean_squared_error(yTest,y_pred))
print("${:,.02f}".format(err))


# In[56]:


test = pd.DataFrame({"Actual": yTest, "Predicted": y_pred})
test


# In[57]:


regressor.score(xTest,yTest)


# In[58]:


x = np.array([["Italy","Master's degree", 20]])
x


# In[59]:


x[:,0] = Country_LE.transform(x[:,0])
x[:,1] = EdLevel_LE.transform(x[:,1])
x = x.astype(float)
x


# In[60]:


y_pred = regressor.predict(x)
y_pred


# In[61]:


import pickle


# In[62]:


data = {"model": regressor, "Country_LE":Country_LE, "EdLevel_LE":EdLevel_LE}
with open("SavedModel.pkl","wb") as file:
    pickle.dump(data, file)


# In[63]:


with open("SavedModel.pkl", "rb") as file:
    data = pickle.load(file)
    
LoadedRegressor = data["model"]
CountryEncoder = data["Country_LE"]
EdLevelEncoder = data["EdLevel_LE"]


# In[64]:


y_pred = LoadedRegressor.predict(x)
y_pred


# In[65]:


import session_info

session_info.show()


# In[ ]:




