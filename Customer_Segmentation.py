#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df = pd.read_csv("AMS325_marketing_campaign.csv")
df


# In[5]:


df.info()


# In[8]:


df = df.dropna()
df


# In[12]:


df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
dates = []
for i in df["Dt_Customer"]:
    i = i.date()
    dates.append(i)
print("The newest customer's shopping date:", max(dates))
print("The oldest customer's shopping date:", min(dates))


# In[16]:


#Created a feature "Customer_For"
days = []
max_dates = max(dates)
for i in dates:
    diff = max_dates - i
    days.append(diff)

df["Customer_For"] = days
df["Customer_For"] = pd.to_numeric(df["Customer_For"], errors="coerce")
df.head()


# In[18]:


print("Categories in the Marital_Status feature:\n", df["Marital_Status"].value_counts())
print("\nCategories in the Education feature:\n", df["Education"].value_counts())

