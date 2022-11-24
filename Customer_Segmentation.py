#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd


# In[102]:


df = pd.read_csv("AMS325_marketing_campaign.csv")
df


# In[103]:


df.info()


# In[104]:


df = df.dropna()
df


# In[105]:


df2 = df.groupby("ID").sum().sort_values("Income",ascending=False)
df2


# In[106]:


df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
dates = []
for i in df["Dt_Customer"]:
    i = i.date()
    dates.append(i)
print("The newest customer's shopping date:", max(dates))
print("The oldest customer's shopping date:", min(dates))


# In[107]:


#Created a feature "Customer_For"
days = []
max_dates = max(dates)
for i in dates:
    diff = max_dates - i
    days.append(diff)
df["Customer_For"] = days
df["Customer_For"] = pd.to_numeric(df["Customer_For"], errors="coerce")
df


# In[108]:


print("Categories in the Marital_Status feature:\n", df["Marital_Status"].value_counts())
print("\nCategories in the Education feature:\n", df["Education"].value_counts())


# ### Feature Engineering 

# In[109]:


# Change Year_Birth to Age 
df["Age"] = 2022 - df["Year_Birth"]

# Change Education feature into 3 categories (Undergraduate & Graduate & Postgraduate)
edu = {'Basic':'Undergraduate','2n Cycle':'Postgraduate','Master':'Postgraduate','PhD':'Postgraduate','Graduation':'Graduate'}
df["Education"] = df["Education"].replace(edu)

# Create Live_With feature that converts Marital_Status into 2 cateogries (Couple & Single)
mar_stat = {'Married':'Together', 'Divorced':'Single', 'Widow':'Single', 'Alone':'Single', 'Absurd':'Single', 'YOLO':'Single'}
df["Live_With"] = df["Marital_Status"].replace(mar_stat)

# Create Children_Home feature that combines Kidhome & Teenhome 
df["Children_Home"] = df["Kidhome"] + df["Teenhome"] 

# Create Family_Num feature that displays the size of the family 
n_size = {"Single": 1, "Together": 2}
df["Family_Size"] = df["Live_With"].replace(n_size) + df["Children_Home"]

# Create Spent feature that combines all expenditures of products 
df["Spent"] = df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
df

