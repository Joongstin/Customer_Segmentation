#!/usr/bin/env python
# coding: utf-8

# In[172]:


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


# In[173]:


# load data
df = pd.read_csv("AMS325_marketing_campaign.csv")
df


# In[174]:


# look for information about the dataset
df.info()
df.columns


# In[175]:


# drop missing values
df3 = df.dropna()
df3


# In[176]:


# grouping by ID column to see if there's any overlapping customers in the dataset 
df2 = df3.groupby("ID").sum().sort_values("Income",ascending=False)
df2


# In[177]:


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


# In[178]:


#Create a feature "Customer_For" that indicates the number of days the customer shopped relative to the newest recorded date
days = []
max_dates = max(dates)
for i in dates:
    diff = max_dates - i
    days.append(diff)

df3.loc[:,"Customer_For"] = days
df3.loc[:,"Customer_For"] = pd.to_numeric(df3.loc[:,"Customer_For"], errors="coerce")
df3


# In[179]:


#cat_cols contain categorical variables
cat_cols=['Education', 'Marital_Status', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain']
for column in cat_cols:
    print(df[column].value_counts(normalize=True))
    print("-" * 40)


# ### Feature Engineering 

# In[180]:


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


# In[181]:


# Display the general statistics for all features 
df3.describe()


# In[182]:


# Create pair plot graphs that display relationship among Income, Recemcy. Customer_For, Age, Spent features  
sns.set(style='whitegrid', context='notebook')
cols = ["Income", "Recency", "Customer_For", "Age", "Spent", "IsParent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df3[cols], hue= "IsParent", markers=["o","s"])
plt.show()


# In[183]:


# Remove outliers for Income and Age features 
cond = df3.loc[:,"Income"] < 600000
df3 = df3.loc[cond]

cond2 = df3.loc[:,"Age"] < 110
df3 = df3.loc[cond2]
print("Total length of dataset after removing outliers:", len(df3))


# In[184]:


# Display pair plot graphs after removing outliers
sns.set(style='whitegrid', context='notebook')
cols = ["Income", "Recency", "Customer_For", "Age", "Spent", "IsParent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df3[cols], hue= "IsParent", markers=["o","s"])
plt.show()


# In[185]:


# Create the correlation matrix 
matrix = df3.corr().round(2)
plt.figure(figsize=(22,22))
sns.heatmap(matrix, annot=True, center=0, cmap="coolwarm")
plt.show()


# In[186]:


Camp_cols=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

success_campaign=(df[Camp_cols].sum()/df.shape[0])*100
print(success_campaign)
# plot
success_campaign.plot(kind='bar', figsize=(6,6))
plt.ylabel("Perentage")
plt.show()


# In[211]:


col_list = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases','NumWebVisitsMonth']
plt.figure(figsize=(15,15))
for i in range(len(col_list)):
    p1=pd.DataFrame(df3.groupby(['AcceptedCmp1']).mean()[col_list[i]]).T
    p2=pd.DataFrame(df3.groupby(['AcceptedCmp2']).mean()[col_list[i]]).T
    p3=pd.DataFrame(df3.groupby(['AcceptedCmp3']).mean()[col_list[i]]).T
    p4=pd.DataFrame(df3.groupby(['AcceptedCmp4']).mean()[col_list[i]]).T
    p5=pd.DataFrame(df3.groupby(['AcceptedCmp5']).mean()[col_list[i]]).T
    plt.subplot(2,2 ,i+1)
    con = pd.concat([p1,p2,p3,p4,p5],axis=0).set_index([Camp_cols])
    plt.plot(con)
    plt.ylabel('Average amount spend on' + ' ' + col_list[i])


# In[210]:


col_list = ["Wines", "Fruits", "Meat", "Fish", "Sweet", "Gold"]
plt.figure(figsize=(15,15))
for i in range(len(col_list)):
    p1=pd.DataFrame(df3.groupby(['AcceptedCmp1']).mean()[col_list[i]]).T
    p2=pd.DataFrame(df3.groupby(['AcceptedCmp2']).mean()[col_list[i]]).T
    p3=pd.DataFrame(df3.groupby(['AcceptedCmp3']).mean()[col_list[i]]).T
    p4=pd.DataFrame(df3.groupby(['AcceptedCmp4']).mean()[col_list[i]]).T
    p5=pd.DataFrame(df3.groupby(['AcceptedCmp5']).mean()[col_list[i]]).T
    plt.subplot(3,2 ,i+1)
    con = pd.concat([p1,p2,p3,p4,p5],axis=0).set_index([Camp_cols])
    plt.plot(con)
    plt.ylabel('Average amount spend on' + ' ' + col_list[i])
 


# In[212]:


#Recency
def Purchases_per_campaign(columns_name):
    dp1=pd.DataFrame(df3.groupby(['AcceptedCmp1']).mean()[columns_name]).T
    dp2=pd.DataFrame(df3.groupby(['AcceptedCmp2']).mean()[columns_name]).T
    dp3=pd.DataFrame(df3.groupby(['AcceptedCmp3']).mean()[columns_name]).T
    dp4=pd.DataFrame(df3.groupby(['AcceptedCmp4']).mean()[columns_name]).T
    dp5=pd.DataFrame(df3.groupby(['AcceptedCmp5']).mean()[columns_name]).T
    # dp6=pd.DataFrame(df.groupby(['AcceptedCmp6']).mean()[columns_name]).T
    pd.concat([dp1,dp2,dp3,dp4,dp5],axis=0).set_index([Camp_cols]).plot(kind='line', figsize=(8,8))
    plt.ylabel('Average' + ' ' + columns_name)
    plt.show()

Purchases_per_campaign('Recency')


# In[190]:


# Find the categorical variables in orginal data (df3) and change it to the numerical variable by using LabelEncoder.
from sklearn.preprocessing import LabelEncoder
s = (df3.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:", object_cols)

LE=LabelEncoder()
for i in object_cols:
    df3[i]=df3[[i]].apply(LE.fit_transform)
print("All features are now numerical")


# In[191]:


# The data which we changed the categorical variable to numerical.
df3


# In[192]:


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


# In[193]:


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
    ax.annotate(r"%s" % ((str(vals[i]*100)[:3])), (ind[i], vals[i]), 
    va = "bottom", ha = "center", fontsize = 10)
 
ax.set_xlabel("PC")
ax.set_ylabel("Variance")
plt.title('Scree plot')


# In[194]:


principalComponents   = principalComponents[:,0:3]
principalDf = pd.DataFrame(data=principalComponents, columns=[
                           'Principal_Component_1', 'Principal_Component_2','Principal_Component_3'])

principalDf.describe().T


# In[195]:


principalDf


# In[196]:


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


# In[197]:


# Using the Elbow method, we can find the optimal number of cluster, in this case is 4. 
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

Elbow_Method = KElbowVisualizer(KMeans(), k=10)
Elbow_Method.fit(principalDf)
Elbow_Method.show()


# In[198]:


# We will perform the K means cluster to find the customer segmenatation.
kmeans = KMeans(n_clusters=4, random_state=1)
kmeans.fit(principalDf)
clusters = kmeans.fit_predict(principalDf)
principalDf["Cluster"] = clusters
df3["Cluster"] = clusters


# In[199]:


principalDf


# In[200]:


df3


# In[201]:


# Coloring each dot depends on the cluster. 
from matplotlib import colors
cmap = colors.ListedColormap(["#1f77b4","#d62728","#9467bd","#2ca02c"])
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=20, c=principalDf["Cluster"], marker='o', cmap =cmap )
ax.set_title("The Plot Of The Clusters")
plt.show()


# In[202]:


# Find the distribution of the cluster.
col = ["#1f77b4","#d62728","#9467bd","#2ca02c"]
pl = sns.countplot(x=df3["Cluster"], palette= col)
pl.set_title("Distribution Of The Clusters")
plt.show()


# In[203]:


#The clusters seem to be fairly distributed.
pl = sns.scatterplot(data = df3,x=df3["Spent"], y=df3["Income"],hue=df3["Cluster"], palette= col,s=20)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# In[204]:


print("Income mean:", df3['Income'].mean())
print("Income mean by cluster:")
print("Group 0:",df3[df3["Cluster"] == 0].Income.mean())
print("Group 1:",df3[df3["Cluster"] == 1].Income.mean())
print("Group 2:",df3[df3["Cluster"] == 2].Income.mean())
print("Group 3:",df3[df3["Cluster"] == 3].Income.mean())

print("\nSpent mean:",df3['Spent'].mean())
print("Spent mean by cluster:")
print("Group 0:",df3[df3["Cluster"] == 0].Spent.mean())
print("Group 1:",df3[df3["Cluster"] == 1].Spent.mean())
print("Group 2:",df3[df3["Cluster"] == 2].Spent.mean())
print("Group 3:",df3[df3["Cluster"] == 3].Spent.mean())


# Income vs spending plot shows the clusters pattern
# 
# - group 0: very low spending & very low income 
# - group 1: very high spending & very high income
# - group 2: high spending & high income
# - group 3: low spending & low income

# In[205]:


#We can explore what each cluster is spending on for the targeted marketing strategies.
#Creating a feature to get a sum of accepted promotions 
df3["Total_Promos"] = df3["AcceptedCmp1"]+ df3["AcceptedCmp2"]+ df3["AcceptedCmp3"]+ df3["AcceptedCmp4"]+ df3["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=df3["Total_Promos"],hue=df3["Cluster"], palette= col)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()


# In[206]:


# making dataframes of customers having income <52k and >52K
df_1=df3[df3.Cluster==0]
df_2=df3[df3.Cluster==1]
df_3=df3[df3.Cluster==2]
df_4=df3[df3.Cluster==3]


Camp_cols=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

#Calculating success rate of each campaing for both segments 
success_campaign1=pd.DataFrame((df_1[Camp_cols].sum()/df_1.shape[0])*100, columns=['Group 0: Very Low Income & Very Low Spent'])

success_campaign2=pd.DataFrame((df_2[Camp_cols].sum()/df_2.shape[0])*100, columns=['Group 1: Low Income & Low Spent'])

success_campaign3=pd.DataFrame((df_3[Camp_cols].sum()/df_3.shape[0])*100, columns=['Group 2: High Income & High Spent'])

success_campaign4=pd.DataFrame((df_4[Camp_cols].sum()/df_4.shape[0])*100, columns=['Group 3: Very High Income & Very High Spent'])


new_df=pd.concat([success_campaign1, success_campaign2,success_campaign3,success_campaign4], axis=1)

# plot
plt.figure(figsize=(8,8))
sns.lineplot(data=new_df)
plt.title("Percentage Acceptance of each campaign")
plt.ylabel("Percentage Acceptance of a campaign")
plt.show()


# In[207]:


plt.figure()
pl=sns.boxenplot(y=df3["NumDealsPurchases"],x=df3["Cluster"], palette= col)
pl.set_title("Number of Deals Purchased")
plt.show()


# In[208]:


Personal = ["Children_Home", "Age", "Family_Size", "IsParent", "Education","Live_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=df3[i], y=df3["Spent"], hue =df3["Cluster"], kind="kde", palette=col)
    plt.show()

