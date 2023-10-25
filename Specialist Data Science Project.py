#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


pro_df=pd.read_csv(r'C:\Users\danie\OneDrive\Documents\Py_Project\PgaTourData.csv')
pro_df.describe()


# In[5]:


pro_df.head()


# In[6]:


# sorting the list in descending order
pro_df.sort_values("POINTS")


# In[10]:


# Import the matplotlib.pyplot and name it plt
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# Plot a histogram of "AVG_SCORE" for pro_df
ax.hist(pro_df["AVG_SCORE"])


# In[12]:


# Compare  histogram of "Average Score" for Scratch_data
ax.hist(Scratch_data["AVG_SCORE"])


# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# Plot a histogram of "AVG_Score" for pro_df
ax.hist(pro_df["AVG_SCORE"])

# Compare  histogram of "AVG_Score" for Scratch_data
ax.hist(Scratch_data["AVG_SCORE"])

# Set the x-axis label to "AVG_Score"
ax.set_xlabel("AVG_SCORE")

# Set the y-axis label to "differences"
ax.set_ylabel("HOLES_PLAYED")

plt.show()


# In[16]:


fig, ax = plt.subplots()

# using Fairway Percentage and AVG_CARRY_DISTANCE as x-y, index as color
ax.scatter(pro_df["AGE"], pro_df["AVG_CARRY_DISTANCE"], c=pro_df.index)

# Set the x-axis label to "AGE"
ax.set_xlabel("AGE")

# Set the y-axis label to "AVG_CARRY_DISTANCE"
ax.set_ylabel("AVG_CARRY_DISTANCE")

plt.show()


# In[17]:


# Drop duplicates
pro_df.drop_duplicates()


# In[19]:


# get the number of missing data points per column
missing_values_count = pro_df.isnull().sum()
missing_values_count


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


fig, ax = plt.subplots()
# Plot a histogram of "AVG_SCORE" for pro_df
ax.hist(pro_df["AVG_SCORE"])
plt.show


# In[23]:


pro_df['POINTS_BEHIND_LEAD'] = pd.to_numeric(pro_df['POINTS_BEHIND_LEAD'], errors='coerce').fillna(0).astype(int)
pro_df.dtypes


# In[24]:


pro_df['AGE'].min()


# In[25]:


pro_df['AGE'].max()


# In[26]:


sns.boxplot(data=pro_df,x="AGE")
plt.show


# In[27]:


pro_df.agg(["mean", "std"])


# In[69]:


# Group by Player; calculate MAKES_BOGEY%
pro_by_bogey_percentage = pro_df.groupby("MAKES_BOGEY%")["Player"].sum()
print(pro_by_bogey_percentage)


# In[71]:


sns.barplot(data=pro_df.groupby("MAKES_BOGEY%")["Player"].sum()
, x="ROUNDS_PLAYED", Y= "MAKES_BOGEY%")
plt.show()


# In[30]:


print(pro_df.isna().sum) #No missing Data


# In[33]:


import re

#Check if the string starts with "TOTAL" and ends with "STROKES":

txt = "TOTAL_STROKES"
x = re.search("^TOTAL.*STROKES$", txt)

if x:
  print("YES! We have a match!")
else:
  print("No match")


# In[34]:



txt = "TOTAL_STROKES"

#Return a match at every no-digit character:

x = re.findall("\D", txt)

print(x)

if x:
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[35]:


#Return a list containing every occurrence of "IN":

txt = "POINTS_BEHIND_LEAD"
x = re.findall("IN", txt)
print(x)


# In[36]:



txt = "POINTS_BEHIND_LEAD"

#Check if "ROUND" is in the string:

x = re.findall("ROUND", txt)
print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[37]:


txt = "POINTS_BEHIND_LEAD"
x = re.search("\s", txt)

print("The first white-space character is located in position:", x.start()) 


# In[38]:


#Explain above that code has no space in text (code doesn't run). However, Space in this text(Code runs)
txt = "POINTS_BEHIND_LEAD"
x = re.search("\s", txt)
if (x):
  print("Yes, there is a space")
else:
  print("No space")


# In[39]:


#Replace all white-space characters with the digit "9":

txt = "POINTS_BEHIND_LEAD"
x = re.sub("\s", "9", txt)
print(x)


# In[40]:


#Replace all white-space characters with the digit "9":
txt = "POINTS_ BEHIND_ LEAD"
x = re.sub("\s", "9", txt)
print(x)


# In[41]:


# Using a list as an iterable
my_list = [1, 2, 3, 4, 5]
iterable = iter(my_list)  # Create an iterator from the list

for item in iterable:
    print(item)

    


# In[42]:


my_list = [1, 2, 3, 4, 5]
iterator = iter(my_list)

try:
    while True:
        item = next(iterator)
        print(item)
except StopIteration:
    pass


# In[63]:


Scratch_data = {
    'Player': ["Daniel Devlin"],
    'EVENTS_PLAYED': [10],
    'POINTS': [550],
    'NUMBER_OF_WINS': [1],
    'NUMBER_OF_TOP_Tens': [3],
    'POINTS_BEHIND_LEAD': [2500],
    'ROUNDS_PLAYED': [50],
    'SG_PUTTING_PER_ROUND': [-.003],
    'TOTAL_SG:PUTTING': [-2],
    'MEASURED_ROUNDS': [36],
    'AVG_Driving_DISTANCE': [320],
    'AVG_CARRY_DISTANCE': [300],
    'SHORTEST_CARRY_DISTANCE': [270],
    'AVG_SCORE': [72.467],
    'TOTAL_STROKES': [3600],
    'TOTAL_ROUNDS': [50],
    'MAKES_BOGEY%': [2],
    'BOGEYS_MADE': [18],
    'HOLES_PLAYED': [900],
    'AGE': [26]
    
}




# In[64]:


# Create a data frame from the dictionary
Scratch_data_df = pd.DataFrame(Scratch_data)


# In[65]:


# Display the data frame
print(Scratch_data_df)


# In[66]:


import pandas as pd

# Merge the two data frames using the 'common_column' as the key
merged_df = pd.merge(pro_df, Scratch_data_df)

# Display the merged data frame
print(merged_df)


# In[67]:


# Merging Dataframes
merge_df = pro_df.merge(scratch_df, left_on='AVG_SCORE', right_on='AVG_SCORE', how='left')
print(merge_pga)


# In[72]:


# Numpy functions 1
# Import the numpy package as np
import numpy as np


# In[73]:


# Create a numpy array from pga: np_pro
np_pro = np.array(pro_df)


# In[74]:


# Print out type of np_pro
print(type(np_pro))


# In[82]:


# Numpy functions 2
# Import numpy
import numpy as np

# Create a numpy array from scratch_data : np_scratch
np_Scratch = np.array(Scratch_data_df)

# Print out np_scratch
print(type(np_Scratch)


# In[80]:


# Print out np_scratch
print(type(np_Scratch)


# In[54]:


import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Perform some mathematical operations on the array
result = arr + 10
print(result)


# In[55]:


def calculate_square(x):
    square = x * x
    return square


# In[56]:


result = calculate_square(5)
print(result)  

