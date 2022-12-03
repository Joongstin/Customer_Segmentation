#!/usr/bin/env python
# coding: utf-8

# In[216]:


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


# In[217]:


# load data
df = pd.read_csv("AMS325_marketing_campaign.csv")
df


# In[218]:


# look for information about the dataset
df.info()


# In[219]:


# drop missing values
df3 = df.dropna()
df3


# In[220]:


# grouping by ID column to see if there's any overlapping customers in the dataset 
df2 = df3.groupby("ID").sum().sort_values("Income",ascending=False)
df2


# In[221]:


# Create a feature out of Dt_Customer that indicates the range of days a customer is enrolled in the firm's dataset
# In order to get the values I must check the newest and oldest recorded dates.
df3 = df3.copy()
cust = pd.to_datetime(df3.loc[:,"Dt_Customer"])
dates = []
for i in cust:
    i = i.date()
    dates.append(i)
print("The newest customer's shopping date:", max(dates))
print("The oldest customer's shopping date:", min(dates))


# In[222]:


#Create a feature "Customer_For" that indicates the number of days the customer shopped relative to the newest recorded date
days = []
max_dates = max(dates)
for i in dates:
    diff = max_dates - i
    days.append(diff)

df3.loc[:,"Customer_For"] = days
df3.loc[:,"Customer_For"] = pd.to_numeric(df3.loc[:,"Customer_For"], errors="coerce")
df3


# In[223]:


print("Categories in the Marital_Status feature:\n", df3["Marital_Status"].value_counts())
print("\nCategories in the Education feature:\n", df3["Education"].value_counts())


# ### Feature Engineering 

# In[224]:


# Change Year_Birth to Age 
df3["Age"] = 2022 - df3["Year_Birth"]

# Change Education feature into 3 categories (Undergraduate & Graduate & Postgraduate)
edu = ({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})
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


# In[225]:


# Display the general statistics for all features 
df3.describe()


# In[226]:


# Create pair plot graphs that display relationship among Income, Recemcy. Customer_For, Age, Spent features  
sns.set(style='whitegrid', context='notebook')
cols = ["Income", "Recency", "Customer_For", "Age", "Spent", "IsParent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df3[cols], hue= "IsParent", markers=["o","s"])
plt.show()


# In[227]:


# Remove outliers for Income and Age features 
cond = df3.loc[:,"Income"] < 600000
df3 = df3.loc[cond]

cond2 = df3.loc[:,"Age"] < 110
df3 = df3.loc[cond2]
print("Total length of dataset after removing outliers:", len(df3))


# In[228]:


# Display pair plot graphs after removing outliers
sns.set(style='whitegrid', context='notebook')
cols = ["Income", "Recency", "Customer_For", "Age", "Spent", "IsParent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df3[cols], hue= "IsParent", markers=["o","s"])
plt.show()


# In[229]:


# Create the correlation matrix 
matrix = df3.corr().round(2)
plt.figure(figsize=(22,22))
sns.heatmap(matrix, annot=True, center=0, cmap="coolwarm")
plt.show()


# In[230]:


# Find the categorical variables in orginal data (df3) and change it to the numerical variable by using LabelEncoder.
from sklearn.preprocessing import LabelEncoder
s = (df3.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:", object_cols)

LE=LabelEncoder()
for i in object_cols:
    df3[i]=df3[[i]].apply(LE.fit_transform)
print("All features are now numerical")


# In[231]:


# The data which we changed the categorical variable to numerical.
df3


# In[232]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

pca_df = df3.copy()
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
pca_df = pca_df.drop(cols_del, axis=1)
attributes = list(pca_df)
x = pca_df.loc[:, attributes].values
x1 = StandardScaler().fit_transform(x)
x1_df = pd.DataFrame(x1,columns= pca_df.columns )
x1_df.head()


# In[233]:


pca = PCA(n_components=8)
principalComponents = pca.fit_transform(x1)
num_components = len(pca.explained_variance_ratio_)
ind = np.arange(num_components)
vals = pca.explained_variance_ratio_ 
    
ax = plt.subplot()
cumvals = np.cumsum(vals)
ax.bar(ind, vals)
ax.plot(ind, cumvals, color = 'red') 

for i in range(num_components): 
    ax.annotate(r"%s" % ((str(vals[i]*100)[:3])), (ind[i], vals[i]), va = "bottom", ha = "center", fontsize = 10)
 
ax.set_xlabel("PC")
ax.set_ylabel("Variance")
plt.title('Scree plot')


# In[234]:


principalComponents   = principalComponents[:,0:3]
principalDf = pd.DataFrame(data=principalComponents, columns=[
                           'Principal_Component_1', 'Principal_Component_2','Principal_Component_3'])

principalDf.describe().T


# In[235]:


principalDf


# In[236]:


#A 3D Projection Of Data In The Reduced Dimension
x =principalDf["Principal_Component_1"]
y =principalDf["Principal_Component_2"]
z =principalDf["Principal_Component_3"]

#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x,y,z, marker="o" )
ax.set_title("A 3D Projection Of Data")
plt.show()


# In[237]:


# Using the Elbow method, we can find the optimal number of cluster, in this case is 4. 
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

Elbow_Method = KElbowVisualizer(KMeans(), k=10)
Elbow_Method.fit(principalDf)
Elbow_Method.show()


# In[238]:


# We will perform the K means cluster to find the customer segmenatation.
kmeans = KMeans(n_clusters=4, random_state=1)
kmeans.fit(principalDf)
clusters = kmeans.fit_predict(principalDf)
principalDf["Cluster"] = clusters
df3["Cluster"] = clusters


# In[239]:


principalDf


# In[240]:


df3


# In[241]:


# Coloring each dot depends on the cluster. 
from matplotlib import colors
cmap = colors.ListedColormap(["#1f77b4","#d62728","#9467bd","#2ca02c"])
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=20, c=principalDf["Cluster"], marker='o', cmap =cmap )
ax.set_title("The Plot Of The Clusters")
plt.show()


# In[242]:


# Find the distribution of the cluster.
col = ["#1f77b4","#d62728","#9467bd","#2ca02c"]
pl = sns.countplot(x=df3["Cluster"], palette= col)
pl.set_title("Distribution Of The Clusters")
plt.show()

