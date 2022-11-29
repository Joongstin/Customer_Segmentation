#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import cluster
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(555)


# In[16]:


df = pd.read_csv("AMS325_marketing_campaign.csv")
df


# In[17]:


df.info()


# In[18]:


df = df.dropna()
df


# In[19]:


df2 = df.groupby("ID").sum().sort_values("Income",ascending=False)
df2


# In[20]:


df3 = df.copy()
cust = pd.to_datetime(df3.loc[:,"Dt_Customer"])
dates = []
for i in cust:
    i = i.date()
    dates.append(i)
print("The newest customer's shopping date:", max(dates))
print("The oldest customer's shopping date:", min(dates))


# In[21]:


#Created a feature "Customer_For"
days = []
max_dates = max(dates)
for i in dates:
    diff = max_dates - i
    days.append(diff)

df3.loc[:,"Customer_For"] = days
df3.loc[:,"Customer_For"] = pd.to_numeric(df3.loc[:,"Customer_For"], errors="coerce")
df3


# In[22]:


print("Categories in the Marital_Status feature:\n", df3["Marital_Status"].value_counts())
print("\nCategories in the Education feature:\n", df3["Education"].value_counts())


# ### Feature Engineering 

# In[23]:


# Change Year_Birth to Age 
df3["Age"] = 2022 - df3["Year_Birth"]

# Change Education feature into 3 categories (Undergraduate & Graduate & Postgraduate)
edu = {'Basic':'Undergraduate','2n Cycle':'Postgraduate','Master':'Postgraduate','PhD':'Postgraduate','Graduation':'Graduate'}
df3["Education"] = df3["Education"].replace(edu)

# Create Live_With feature that converts Marital_Status into 2 cateogries (Couple & Single)
mar_stat = {'Married':'Together', 'Divorced':'Single', 'Widow':'Single', 'Alone':'Single', 'Absurd':'Single', 'YOLO':'Single'}
df3["Live_With"] = df3["Marital_Status"].replace(mar_stat)

# Create Children_Home feature that combines Kidhome & Teenhome 
df3["Children_Home"] = df3["Kidhome"] + df3["Teenhome"] 

# Create Is_Parent feature that displays the parenthood status
cond = np.where(df3.Children_Home > 0,1,0)
df3["IsParent"] = cond

# Create Family_Num feature that displays the size of the family 
n_size = {"Single": 1, "Together": 2}
df3["Family_Size"] = df3["Live_With"].replace(n_size) + df3["Children_Home"]

# Create Spent feature that combines all expenditures of products 
df3["Spent"] = df3["MntWines"] + df3["MntFruits"] + df3["MntMeatProducts"] + df3["MntFishProducts"] + df3["MntSweetProducts"] + df3["MntGoldProds"]

# rename some food features for clarity 
food_name = {"MntWines":"Wines", "MntFruits":"Fruits", "MntMeatProducts":"Meat", "MntFishProducts":"Fish","MntSweetProducts":"Sweet","MntGoldProds":"Gold"}
df3 = df3.rename(columns=food_name)

# eliminate redundant/unnecessary features 
rep_features = ["ID", "Year_Birth", "Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue"]
df3 = df3.drop(rep_features, axis='columns')
df3


# In[24]:


df3.describe()


# In[33]:


sns.set(style='whitegrid', context='notebook')
cols = ["Income", "Recency", "Customer_For", "Age", "Spent", "IsParent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df3[cols], hue= "IsParent", markers=["o","s"])
plt.show()


# In[41]:


# Removing outliers 
cond = df3.loc[:,"Income"] < 600000
df3 = df3.loc[cond]

cond2 = df3.loc[:,"Age"] < 110
df3 = df3.loc[cond2]
print("Total length of dataset after removing outliers:", len(df3))


# In[42]:


sns.set(style='whitegrid', context='notebook')
cols = ["Income", "Recency", "Customer_For", "Age", "Spent", "IsParent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df3[cols], hue= "IsParent", markers=["o","s"])
plt.show()


# In[46]:


# Create the correlation matrix 
matrix = df3.corr().round(2)
plt.figure(figsize=(22,22))
sns.heatmap(matrix, annot=True, center=0)
plt.show()

