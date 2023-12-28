# %% [markdown]
## Import Libraries
# %%
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
sns.set_palette('ocean_r')
from sklearn.model_selection import train_test_split
from currency_converter import CurrencyConverter
from sklearn.linear_model import LinearRegression, SGDRegressor
from lightgbm import LGBMRegressor, plot_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from skopt  import BayesSearchCV # pip install scikit-optimize
import pickle
 # %% [markdown]
## Import Data
# %%
# Uncomment, run this just once, and then, comment out the code in this cell.
# od.download("https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps")


# %%
df = pd.read_csv("google-playstore-apps/Google-Playstore.csv")
df.head()

# %% [markdown]
## Data Exploration
# %%
df.shape
# %%
df.columns
# %%
df.describe()
# %%
df.describe(include='O')
# %% [markdown]
# We notice the count of app name and unique names doesn't match, so either there are duplicated values or missing values.
# But we see app ID having all unique values. Let's see.
# %%
df.isna().sum()
# %% [markdown]
# So, there are 5 app names missing but all the app IDs are present.
# %%
df[df['App Name'].isna()]
# %%
df['App Name'].duplicated().sum()
# %%
# %%
df['App Id'].isna().sum()
# %%
df['App Id'].duplicated().sum()
# %%
df[df['App Name'].duplicated()][['App Name', 'App Id']].head(10)

# %% [markdown]
## Handling Missing values 
# So, we notice that there are no missing app IDs and none of the app IDs are duplicated as well.
# There are multiple apps with same names but the ID is different, so it's fine.
# App Id, will thus be used as our primary key. 
# The 5 records in which the app name are missing, we'll remove those.

# %%
df_clean = df.copy()
df_clean = df_clean.dropna(subset=['App Name'])
df_clean.isna().sum()
# %% [markdown]
# There are 22883 apps that do not have a rating or a rating count. 
# Our ultimate target is to predict rating of an app. So imputing the missing values, no matter how good, isn't original. 
# With 2+ million data, we can afford to drop 22k records. So, let's do that.
# %%
df_clean = df_clean.dropna(subset=['Rating'])
df_clean = df_clean.dropna(subset=['Rating Count'])
# %%
print(df_clean[['Installs', 'Minimum Installs']].dtypes)
df_clean[['Installs', 'Minimum Installs']].head()
# %% [markdown]
# The values in minimum installs are object type with '+' appended at the end. 
# It also has commas.
# It also seems like the values in the 2 columns are same.
# Let's remove those and have Installs as a numerical column and then compare 
# to check if the values are actually the same.
# %%
df_clean['Installs']=df_clean['Installs'].map(lambda x: x[:-1]) # Removing the '+'
df_clean['Installs']= df_clean['Installs'].map(lambda x: x.replace(',', '')) # Removing the ','
df_clean['Installs']= pd.to_numeric(df_clean['Installs'])
df_clean['Installs'].equals(df_clean['Minimum Installs'].astype('int64'))
# %% [markdown]
# So both the columns have same values. The meaning of both the features is also more or less the same.
# So, we will drop one of them.
# %%
df_clean = df_clean.drop('Installs', axis=1)
df_clean.isna().sum()
# %%[markdown]
# Now let's concentrate on the currency column which has 20 nan values. We'll just remove these.
# %%
df_clean = df_clean.dropna(subset=['Currency'])
df_clean.isna().sum()
# %%[markdown]
# Since we have 6526 rows in Minimum Android consisting of nan we can afford
# to drop it .Considering the amount of data we have it should not possess a problem
# %%
df_clean = df_clean.dropna(subset=['Minimum Android'])
df_clean.isna().sum()
# %%[markdown]
# `Developer ID` and `Developer Email` are useful information but unnecessary for our model, so we don't need to deal with their missing values 
# and we will drop those columns going ahead.
# 
# `Developer Website` has 751356 NA values and so we have chosen not to drop those since it might result in significant data loss.
# Going ahead we will use this column for feature engineering and then drop this column


# %%[markdown]
# We notice that Released and privacy policy have too much NA values. 
# Released column has 48371 NA values. We'll impute them using median.
# But, we'll be using these for analysis and feature engineering going ahead.
# So for now, we'll keep the missing values since it doesn't pose an issue.
# %%
# Imputing the missing values of released column  with median date 
df_clean['Released'] = pd.to_datetime(df_clean['Released'], errors='coerce')
median_date = df_clean['Released'].median()  # Calculate the median date
df_clean['Released'].fillna(median_date, inplace=True)
df_clean['Released'].isna().sum() #rechecking for missing values

# %%[markdown]
## EDA - Exploratory Data Analysis
### Data Cleaning & Feature Engineering
# %%
print(df_clean['Minimum Android'].value_counts())
# %%[markdown]
# This has values in the form of '4.1 and up'. Let's clean it by removing the strings and rounding up 
# to get the float minimum android version

# %%
# TODO: Have separate ideas for for 'and up' and versions b/w 2.
# Function to extract the numeric part, round up, and return the first three characters
def extract_and_round_up(version_string):
    try:
        # Split the string, take the first part, convert to float, round up, and return the first three characters
        # The basic reason of applying ceiling function is because
        return str(math.ceil(float(version_string.split()[0][:3])))
    except (ValueError, IndexError):
        # Return the original string in case of an exception
        return version_string

df_clean['Minimum Android'] = df_clean['Minimum Android'].apply(extract_and_round_up)
print(df_clean['Minimum Android'])
 
# %%
#Visualising Minimum android Column
plt.figure(figsize=(10, 6))
df_clean['Minimum Android'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Minimum Android Versions')
plt.xlabel('Minimum Android Version')
plt.ylabel('Count')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()
# %%[markdown]
# * Android Version 5:
# This version appears most frequently in the dataset, with a count
# of (please enter number as data will be changed), indicating a significant presence of apps designed for
# Android version 5.
# * Android Version 4: The second most common version, appearing
# 338,684 times.
# * Varies with Device: Indicates cases where the minimum Android
# version is flexible or unspecified, occurring 24,322 times. 

# %%
# We can't work with the varies with device, so we'll remove those
df_clean = df_clean[df_clean['Minimum Android']!= 'Varies with device']

# %%
currency_counts = df_clean['Currency'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(currency_counts, labels=currency_counts.index, autopct='%1.2f%%', startangle=90)
plt.title('Distribution of Currencies')
plt.legend(currency_counts.index, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
# %%[markdown]
### Conclusion:
# The dataset is heavily dominated by transactions in U.S. Dollars, as evidenced by the high probability associated with USD.
# The low probabilities for other currencies suggest that these alternate currencies 
# are rare or infrequently represented in the dataset. 
# The presence of 'XXX' still indicates some instances where 
# the currency information is unspecified or missing, albeit at a very low probability.

# %%
df_clean['Currency'].value_counts(normalize=True)
# %%[markdown]
# USD (U.S. Dollar): The probability of encountering the U.S. Dollar in the dataset is extremely high, at approximately 99.946%. 
# This suggests that the overwhelming majority of entries in the dataset are denominated in U.S. Dollars.
# XXX (Unknown Currency): This has a probability of approximately
# 0.053%, indicating that there are a small number of instances
# where the currency information is either missing or not specified.
# The rest of the currencies have very low probabilities (in the range of 0.0000026% to 0.000000044%) 
# relative to the U.S. Dollar, indicating their infrequent occurrence in the dataset.
# 
# We'll leave these untouched here, and during preprocessing we'll convert all the currency to USD

# %%
# Preprocessing the size column
print(df_clean['Size'].unique())
# %%[markdown]
# So, the app sizes are in GB, MB, and KB. Let's convert all those in to KBs 
# %%
# Getting the count of apps of various sizes 
countm_M=0
countk_K=0
countg_G=0
count_varieswithdevice_nan =0
for values in df_clean['Size']:
    if 'M' in str(values) or 'm' in str(values):
        countm_M+=1
    elif 'K' in str(values) or 'k' in str(values):
        countk_K+=1
    elif 'Varies with device' in str(values) or str(values)=='nan':
        count_varieswithdevice_nan+=1
    elif 'G' in str(values) or 'g' in str(values):
        countg_G+=1


total_count=countm_M+countk_K+countg_G+count_varieswithdevice_nan
print(total_count)
print(len(df_clean['Size']))

# %% 
# The various sizes of apps are listed down below
print(f'Apps of size in megabytes are {countm_M}')
print(f'Apps of size in kilobytes are {countk_K}')
print(f'Apps of size in gigabytes are {countg_G}')
print(f'Apps of variable sizes and also of nan values are {count_varieswithdevice_nan}')

# %%
# Here we convert apps of sizeM(megabytes) 
# into their corresponding values in kilobytes(k)
def convert_m_to_kb(x):
    if 'M' in x or 'm' in x:
        return pd.to_numeric(x.replace('M', '').replace('m', '').replace(',','')) * 1024
    else:
        return x

# Convert 'M' or 'm' to kilobytes
df_clean['Size'] = df_clean['Size'].astype(str).apply(convert_m_to_kb)
x = (df_clean['Size'] == 'm') | (df_clean['Size'] == 'M')
# Once we have created the boolean mask x, you can use x.sum() 
# to count the number of True values in the mask. 
# In the context of above conversion, x.sum() would give us
# the total count of rows where the 'size' column is either 'm' or 'M'
# But since we have converted it should give us 0.
count_of_m_or_M = x.sum()
print(f"Count of 'm' or 'M' in the 'size' column: {count_of_m_or_M}")
df_clean['Size']


# %%
# Here we convert apps of size k(kilobytes) into 
# numeric value of the given kilobytes(k) in the datframe
def convert_k_to_numeric(x):
    try:
        if 'K' in x or 'k' in x:
            return pd.to_numeric(x.replace('K', '').replace('k', '').replace(',',''))
    except:
        print(x)
    return x
    
# Convert 'K' or 'k' to numeric value of kilobytes
df_clean['Size'] = df_clean['Size'].astype(str).apply(convert_k_to_numeric)
y = (df_clean['Size'] == 'k') | (df_clean['Size'] == 'K')
# Once we have created the boolean mask y, we can use y.sum() 
# to count the number of True values in the mask. 
# In the context of above conversion, y.sum() would give us
# the total count of rows where the 'size' column is either 'k' or 'K'
# But since we have converted it should give us 0.
count_of_k_or_K = y.sum()
print(f"Count of 'k' or 'K' in the 'size' column: {count_of_k_or_K}")
df_clean['Size']

# %%
# Here we convert apps of size G(Gigabytes) 
# into their corresponding values in kilobytes(k)
def convert_g_to_numeric(x):
    if 'G' in x or 'g' in x:
        return pd.to_numeric(x.replace('G', '').replace('g', '').replace(',','')) * (1024**2)
    else:
        return x
# Convert 'G' or 'g' to kilobytes
df_clean['Size'] = df_clean['Size'].astype(str).apply(convert_g_to_numeric)
z = (df_clean['Size'] == 'g') | (df_clean['Size'] == 'G')
# Once we have created the boolean mask z, we can use z.sum() 
# to count the number of True values in the mask. 
# In the context of above conversion, z.sum() would give us
# the total count of rows where the 'size' column is either 'g' or 'G'
# But since we have converted it should give us 0.
count_of_g_or_G = z.sum()
print(f"Count of 'g' or 'G' in the 'size' column: {count_of_g_or_G}")
df_clean['Size']

# %%
varieswithdevice_nan=0
for values in df_clean['Size']:
    if str(values) =='Varies with device' or str(values)=='nan':
        varieswithdevice_nan += 1
# After preprocessing , cleaning and converting the above rows 
# to Kilobytes as the base value I have kept the total no of string 
# values for "varies with device" and "nan" untampered    
if count_varieswithdevice_nan == varieswithdevice_nan:
    print("Unaltered before and after preprocessing")

#The Nan values is zero
print(df_clean['Size'].isna().sum())


# %%
print(df_clean['Category'].nunique())
sns.histplot(data=df_clean, x='Category', kde=True, )
plt.xticks(rotation = 90, size = 6)
plt.show()
# %%[markdown]
# We notice that there are 48 different categories, and education category has the highest count.
# The distribution is very uneven with a right tail.
# There are many categories with very less values.
# Let's take a look at the values once
# %%
df_clean['Category'].value_counts(normalize=True)
# %%[markdown]
# On having a clearer look at the data we notice that there are some categories with minor
# spelling changes which are the same. Ex: 'Education' and 'Educational'.
# Lets clean such categories by combining the values into one.
# %%
df_clean['Category'] = df_clean['Category'].str.replace('Educational', 'Education')
df_clean['Category'] = df_clean['Category'].str.replace('Music & Audio', 'Music')
# %%[markdown]
# We dealth with missing values for Released columns earlier. Now, let's create a feature called
# app age from this.
# %%
# Calculating age of the app, by extracting the release date from the current date
df_clean['Year Released']= df_clean['Released'].dt.year #extracting year, month and day
df_clean['Month Released']= df_clean['Released'].dt.month
df['Day of week Released']= df_clean['Released'].dt.dayofweek

current_date=pd.to_datetime('now')
df_clean['App Age'] = round((current_date - df_clean['Released']).dt.days / 365.25, 2) if pd.__version__ >= '1.1.0' else (current_date - df['Released']).dt.days / 365.25

# %%
# Privacy column
# Imputing na value for easy replacement in further steps
df_clean['Privacy Policy'].fillna('Not Available', inplace = True)
# Creating a binary feature indicating whether the app has a privacy policy or not
df_clean['Has Privacy Policy']= df_clean['Privacy Policy'].apply(lambda x: 1 if x != 'Not Available' else 0)
df_clean['Has Privacy Policy']
# %%
# visualizing distribution of apps with and without privacy policy 
counts = df_clean['Has Privacy Policy'].value_counts()

plt.figure(figsize=(8, 5))
counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribution of Apps with and without Privacy Policies')
plt.xlabel('Has Privacy Policy')
plt.ylabel('Number of Apps')
plt.xticks(rotation=0)  
plt.show()
# %%
df_clean['Content Rating'].value_counts()
# %%[markdown]
# Upon running the value counts function on the Content Rating column, it is observed that
# there are a total of six categories under which the apps have been sub-divided.
# The names of the categories seem to be a bit confusing ('Everyone'/ 'Everyone 10+'), so we'll provide better distinction to each.
# %%
df_clean['Content Rating'] = df_clean['Content Rating'].replace('Mature 17+', '17+')
df_clean['Content Rating'] = df_clean['Content Rating'].replace('Everyone 10+', '10+')
df_clean['Content Rating'] = df_clean['Content Rating'].replace('Adults only 18+', '18+')
# %%
# We will now try to visualize the distribution of apps across different content rating categories
df_clean['Content Rating'].value_counts(normalize=True).plot.barh()
# The bar plot shows that most of the apps are labeled as 'Everyone', and in comparison, apps rated
# as '18+' are almost negligible.
# %%
df_clean['Last Updated'].head()
# The 'Last Updated' column is of object type.
# %%
# We'll now extract the year from the 'Last Updated' column using the 'splice_string' function
# created below.
def splice_string(original_string, start, end=None):
    if end is None:
        return original_string[start:]
    else:
        return original_string[start:end]
# %% 
# The extracted year is stored in a new column, 'Year Last Updated'.
df_clean['Year Last Updated'] = df_clean['Last Updated'].apply(lambda x: splice_string(x,8, ))
# Converting the new column to integer type
df_clean['Year Last Updated'] = df_clean['Year Last Updated'].astype(int)
# The range of this column is 2009 to 2021
print(df_clean['Year Last Updated'].max(), df_clean['Year Last Updated'].min())

# %%
df_clean['Developer Website'].isna().sum()
# There are a lot of NA values in 'Developer Website'
# %%
# We'll now create a separate column that will contain the presence or absence of 'Developer Website'
# in the form of boolean (0 or 1/False or True) values.
df_clean['Has Developer Website'] = df_clean['Developer Website'].notna().astype(int)
df_clean['Has Developer Website'].head()
# %%
# Most apps do have a Developer Website, but the apps without a developer website are not less either
sns.countplot(x="Has Developer Website", data=df_clean, palette="ocean")
plt.xlabel("Has Developer Website")
plt.ylabel("Number of Apps")
plt.title("Number of apps with or without Developer Website")
plt.show()

# %%[markdown]
# Let's convert some of the columns to appropriate data types
# %%
df_clean['Size'] = pd.to_numeric(df_clean['Size'], errors='coerce')
df_clean['Minimum Installs'] = df_clean['Minimum Installs'].astype(float)
df_clean['Maximum Installs'] = df_clean['Maximum Installs'].astype(float)
df_clean['Average Installs'] = df_clean[['Minimum Installs', 'Maximum Installs']].mean(axis=1)
# %%
df_clean['Editors Choice'] = pd.factorize(df_clean['Editors Choice'])[0]
df_clean["Ad Supported"] = pd.factorize(df_clean["Ad Supported"])[0]
df_clean["In App Purchases"] = pd.factorize(df_clean["In App Purchases"])[0]

# %%[markdown]
## Visualization & Analysis
# %%
# Now let's take a look at the top 10 categories
top10cat = ['Education', 'Music', 'Business', 'Tools', 
            'Entertainment', 'Lifestyle', 'Books & Reference',
            'Personalization', 'Health & Fitness', 'Productivity']
df_top10cat= df_clean[df_clean['Category'].isin(top10cat)]
df_top10cat['Category'].value_counts(normalize=True).plot.barh()
plt.title("Proportion of top 10 categories")
plt.xlabel("Proportion")
plt.show()
# %% [markdown]
# So the Education category apps take up about 20% of all the apps.
# %%
# Now let's take a look at the app ratings.
sns.histplot(x='Rating', data=df_clean, bins=20, kde=True)
plt.title("Histogram of Ratings")
plt.show()
# Interestingly, we observe that a huge number of apps have 0 rating.
# Let's 1st get a better visual by omitting those
# %%
sns.histplot(x='Rating', data=df_clean[df_clean['Rating']>0], 
             bins=20, kde=True)
plt.title("Histogram of Ratings (0 ratings excluded)")
plt.show()
# We get a better idea from this that most of the apps have a rating b/w 3.5-5

# %%
# Now let's check the number of ratings.
df_clean['Rating Count'].describe().apply('{:.5f}'.format)
# %%[markdown]
# Whoa!! Too high standard deviation and 75% of data is below 42 
# while the maximum value is greater than 1.3 million.
# Let's 1st try to see the entire plot and then we'll see only the ones
# below 42 to get a better idea
# %%
sns.histplot(x='Rating Count', data=df_clean, bins=20)
plt.title("Histogram of Rating Count")
plt.show()
# %%[markdown]
# Okay!, so we don't even see anything here.
# %%
sns.histplot(x='Rating Count', data=df_clean[df_clean['Rating Count']<42], bins=20)
plt.title("Histogram of Rating Count (< 3rd quantile)")
plt.show()
# This shows that majority of apps don't even get any ratings.
# And just a few get over hundreds and thousands of ratings.
# %%
# The maximum value in the above plot show 1e6, i.e., a million.
# Let's see how many apps have over a million ratings
len(df_clean[df_clean['Rating Count']>1e6])
# %%[markdown]
# Just 829 of 2 million+ apps have over a million ratings.

# %%
# Let's try to see apps from which category have higher ratings,
# and which are the categories that get rated the most.
df_clean[df_clean['Rating']>3.5]['Category'].value_counts().head().plot.barh()
plt.title("Top 5 highest rated apps by Category")
plt.xlabel("Count")
plt.show()
# So the education category has the higest rated apps.
# %%
df_clean[df_clean['Rating Count']>1e6]['Category'].value_counts().head(6).plot.barh()
plt.title("Top 6 apps with the most rating counts by Category")
plt.xlabel("Count")
plt.show()
# %%[markdown]
# * Action apps have the most number of ratings. 
# * 74 of total apps with more than a million reviews belong to action category. These could be the action games which are super popular.
# * Sports and music have the same number of ratings and are in top 5.
# * Tools category also has 54 apps over a million reviews. These could be the productivity tool apps that many people use on a regular basis.
# 
# Lets just see some of these apps.
# %%
df_clean[df_clean['Rating Count']>1e6][df_clean['Category']=='Action'][['App Name', 'Rating', 'Rating Count']].head(10)
# %%[markdown]
# As we suspected, it's the most popular action games, such as Shadow Fight 2, Mario Kart, Among Us, etc.

# %%
df_clean['Minimum Installs'].value_counts(normalize=True).plot.barh()
plt.title("Proportion of Minimum installs")
plt.xlabel("Proportion  ")
plt.show()
# On checking the install count, we see that the majority of apps fall 
# in the install range of 10 to 10,000.
# %%
df_clean['Maximum Installs'].describe()
# We don't really get much information from this. 

# %%
# visualization of released column
plt.figure(figsize=(12, 6))
yr = df_clean['Year Released'].value_counts().sort_index()
yr.plot(kind='line', marker='o', color='skyblue')
for x, y in zip(yr.index, yr):
    plt.text(x, y, str(y), ha='right', va='bottom')
plt.title('Trend of App Releases Over the Years')
plt.xlabel('Year Released')
plt.ylabel('Number of Apps')
plt.show()
# %%[markdown]
# We observe a gradual increase in the number of apps over the years and it skyrockets around 2016.
# We see an odd sharp decrease after 2020.(Could be covid)
# %%
# Ad-supported Column
df_clean['Ad Supported'].value_counts()
# Number of apps ad supported are almost the same as that not ad supported

# %%
df_clean['In App Purchases'].isnull().sum()
plt.figure(figsize=(8, 5))
sns.countplot(x='In App Purchases', data=df_clean, palette='viridis',hue= 'In App Purchases',legend= False)
plt.title('Distribution of Apps with and without In-App Purchases')
plt.xlabel('In App Purchases')
plt.ylabel('Number of Apps')
plt.show()
# So Majority of apps do not have in-app purchases
# %%
print('Editor_counts:\n', df_clean['Editors Choice'].value_counts())
# %%
# Visualization of the Editor's choice app
plt.figure(figsize=(8, 5))
sns.countplot(x='Editors Choice', data=df_clean, palette='viridis',legend= False, hue = 'Editors Choice')
plt.yscale('log') # Setting y-axis to log scale for better visualization if needed
plt.title("Distribution of Apps as Editor's Choice or Not")
plt.xlabel("Is Editor's Choice")
plt.ylabel('Number of Apps')
plt.show()

# %%
sns.boxenplot(x="Year Last Updated", data=df_clean, palette="crest", showfliers=True)
plt.ylabel("Year Last Updated")
plt.show()
# sns.boxplot(x="Year Last Updated", data=df_clean, palette="crest")
# plt.ylabel("Year Last Updated")
# plt.show()
# %%[markdown]
# We notice a heavy right skew meaning there's a concentration of updates in the latter years closer to 2020.
# So, there has been a trend of apps being updated more frequently in recent years.
# This could be due to a number of factors, such as the fact that newer apps are more likely to be updated than older apps, 
# or that devs are more likely to update apps that are used frequently.
# %%
sns.stripplot(x='Year Last Updated', y='Rating', data=df_clean, hue='Year Last Updated', legend=False)
plt.title('Year of Last Update vs. Rating')
plt.ylabel('Rating')
plt.xlabel('Year Last Updated')
plt.show()
# Most recently updated apps have a higher rating compared to apps that have been dormant for 
# almost a decade and a half.
# This could be a case related to the fact that most apps were updated recently. 
# %%
# Visualizing the relationship between Content Rating and User Rating via a scatter plot:
sns.stripplot(x='Content Rating', y='Rating', data=df_clean, hue='Content Rating')
plt.title('Content Rating vs. Rating')
plt.ylabel('Rating')
plt.xlabel('Content Rating')
plt.show()
# As we deduced earlier, most apps are made with the 'Everyone' category.
# Also, people using the '18+' and 'Unrated' apps are less likely to leave a rating.

#%%
# Line graph of the update trend:
plt.figure(figsize=(12, 6))
yr = df_clean['Year Last Updated'].value_counts().sort_index()
yr.plot(marker='o', color='#B28EC7')
for x, y in zip(yr.index, yr):
    plt.text(x, y, str(y), ha='right', va='bottom')
plt.xlabel('Year Last Updated')
plt.ylabel('Number of Apps')
plt.title('App Update trend over the years')
plt.tight_layout()
plt.show()
# App updates peaked in 2020. Also this trend is similar to the one we did with Year Released.
# There seems to be a pattern in growing apps post 2016

# %%
sns.barplot(x='Minimum Android', y='Rating', data=df_clean)
plt.title("Rating vs. Minimum Android version")
plt.show()
# It's difficult to notice anything peculiar other than android versions 2 and 3 have higher average ratings
# and higher android versions have relatively lower avg ratings.
# %%
sns.barplot(x='Has Developer Website', y='Rating', data=df_clean)
plt.title("Developer Website availability vs. Rating")
plt.show()
# It shows an app having a developer website has a higher mean rating.

# %%
sns.barplot(x='Free', y='Average Installs', hue="Editors Choice", data=df_clean)
plt.title('Price status vs Average installs by Editors Choice')
plt.show()
# So it is the free apps that have editor's choice that have higher install count. 
# Paid apps have extremely low number of installs which is expected. This indicated that users tend
# to use free apps more.
# %%
sns.barplot(x='Editors Choice', y='Rating', data=df_clean, hue='Editors Choice')
plt.xlabel("Editor's Choice")
plt.ylabel('Rating')
plt.title("Editor's Choice vs. Rating")
plt.show()
# Editor's choice apps have higher ratings as well.
# %%
sns.barplot(x='Editors Choice', y='Average Installs', data=df_clean)
plt.xlabel("Editor's Choice")
plt.ylabel('Average Installs')
plt.title("Editor's Choice vs. Average Installs")
plt.show()
# Interesting! It's like users don't even install the non editor's choice apps
# %%
sns.scatterplot(x= 'Size', y = 'Minimum Installs',data = df_clean, alpha=0.5)
plt.title('Scatter Plot: Size vs. Minimum Installs')
plt.xlabel('Size')
plt.ylabel('Minimum Installs')
plt.show()
# We see patterns hhere, the highest minimum installs are only of apps with lower size.
# For apps with a greater size, minimum installs in very less.
# App size does affect the number of installs.

# %%[markdown]
## Data Preparation
# %%
df_clean.isna().sum()
# Interestingly we notice NAs in size, could be because of 'coerce' when to converted the values to numeric.
# Let's 1st drop those.
# %%
df_clean = df_clean.dropna(subset=['Size', 'Minimum Installs'])
# %%
categorical_columns = df_clean.select_dtypes(include=['object']).columns
numerical_columns = df_clean.select_dtypes(include=['number']).columns
print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

# %%[markdown]
# So, we have a bunch of categorical columns and numerical columns, many of which are not useful.
# Let's remove the unrequired columns and have it as another dataframe. Also, we'll have 2 dataframes, 
# one with the features required for model building and one with the app ID and name. (Will be used during further extension of this project)
# %%

df_final = df_clean.drop(['Developer Website','Developer Email','Developer Id',
                          'Privacy Policy', 'Average Installs','Month Released', 'Year Released',
                          'Year Last Updated', 'Scraped Time', 'Released', 'Last Updated'], axis=1)
# df_clean = df_clean.dropna(subset=['Size', 'Minimum Installs'])
# %%[markdown]
# Now we have 19 features. The currency and price are linked. Let's convert all the currency to one unit- USD.
# And we'll adjust the price accordingly. 
# Once, that is done, we'll drop the currency column, because it will just have 1 value.
# %%
# TODO: Check with Released column for the USD equivalent price at that time.
cc=CurrencyConverter()
def currency_to_USD(data):
    if data not in cc.currencies:
        data=1
    else:
        data=cc.convert(1,data,'USD')
    return data
df_final['Currency']= df_final['Currency'].apply(currency_to_USD)
#  %%
df_final['Price']= df_final['Price']*df_final['Currency']
df_final['Price'].value_counts()
# %%[markdown]
# Now let's have the model data with currency, app id and app name removed
# %%
df_model_data = df_final.drop(['Currency', 'App Name', 'App Id'], axis=1)
df_model_data.head()
# %%[markdown]
# We have Category, Minimum Android, and content rating as categorical columns that need to be encoded.
# Free is a bool type column, so we'll convert it into numeric.
# We have 46 different categories, one-hot encoding it will create too much sparse data. Although it's nominal,
# we'll label encode it first and see how the model performs.
# Minimum Android Version we'll just convert into numerical because, it has numerical values.
# Content Rating we'll label encode
# %%
# Changing data types for input comprehensibility while deployment. Will have to encode these back for model training
df_model_data['Ad Supported'] = df_model_data['Ad Supported'].replace({0: 'No', 1: 'Yes'})
df_model_data['In App Purchases'] = df_model_data['In App Purchases'].replace({0: 'No', 1: 'Yes'})
df_model_data['Editors Choice'] = df_model_data['Editors Choice'].replace({0: 'No', 1: 'Yes'})
df_model_data['Has Privacy Policy'] = df_model_data['Has Privacy Policy'].replace({0: 'No', 1: 'Yes'})
df_model_data['Has Developer Website'] = df_model_data['Has Developer Website'].replace({0: 'No', 1: 'Yes'})
# %%
# df_model_data.to_csv('google-playstore-apps/df_model_data.csv', index=False)
# %%
# df_model_data = pd.read_csv('google-playstore-apps/df_model_data.csv')
# %%
df_model_data['Ad Supported'], _ = pd.factorize(df_model_data['Ad Supported'])
df_model_data['In App Purchases'], _ = pd.factorize(df_model_data['In App Purchases'])
df_model_data['Editors Choice'], _ = pd.factorize(df_model_data['Editors Choice'])
df_model_data['Has Privacy Policy'] = df_model_data['Has Privacy Policy'].replace({'No': 0, 'Yes': 1})
df_model_data['Has Developer Website'] = df_model_data['Has Developer Website'].replace({'No': 0, 'Yes': 1})
# %%
df_model_data['Has Developer Website'] = df_model_data['Has Developer Website'].astype('int32')
df_model_data['Minimum Android'] = df_model_data['Minimum Android'].astype('float64')
# %%
le=LabelEncoder()
df_model_data['Content Rating']=le.fit_transform(df_model_data['Content Rating'])
df_model_data['Category']=le.fit_transform(df_model_data['Category'])
# %%
# Let's take a glimpse of the final data
df_model_data.head()
# %%
plt.figure(figsize=(20,15))
sns.heatmap(df_model_data.corr(),annot=True)
plt.title("Heatmap of the final data")
plt.plot()
plt.show()
# %%
y=df_model_data["Rating"]
x=df_model_data.drop("Rating", axis=1)
train_X,test_X,train_Y,test_Y=train_test_split(x,y,test_size=0.15,random_state=42)
# %%[markdown]
## Model Building
# %%[markdown]
# Since R2 score increases on increasing the number of features, and our data have too many features.
# We will use RMSE to evaluate our regression model. We will use the basic Linear Regression model as
# the baseline. Before this, let's set a seed to get avoid variable results.
# %%
# np.random.seed(7)
# model_lr=LinearRegression()
# model_lr.fit(train_X,train_Y)
# %%
# print('Train RMSE Linear Regression: ', mean_squared_error(train_Y, model_lr.predict(train_X)))
# print('Test RMSE Linear Regression: ',mean_squared_error(test_Y, model_lr.predict(test_X)))
# %%[markdown]
# We get an ~ 3.96 RMSE value in train set and just a bit more, 3.98, on test set.
# Well, good thing that our model isn't overfitting much.
# 
# Now let's try some other models and see their performance
# %%
# np.random.seed(7)
# model_sgd = SGDRegressor()
# model_sgd.fit(train_X, train_Y)
# %%
# print('Train RMSE SGD Regression: ',mean_squared_error(train_Y, model_sgd.predict(train_X)))
# print('Test RMSE SGD Regression: ',mean_squared_error(test_Y, model_sgd.predict(test_X)))
# %%[markdown]
# We get a worse RMSE than the Linear model, it has now increased to 5.41e+45 in train and 5.21e+45 in test.
# Let's try using better models to bring it under 1. 

# %%
np.random.seed(7)
model_lgbm = LGBMRegressor()
model_lgbm.fit(train_X, train_Y)
# %%
print('Train RMSE LGBM Regression: ',mean_squared_error(train_Y, model_lgbm.predict(train_X)))
print('Test RMSE LGBM Regression: ',mean_squared_error(test_Y, model_lgbm.predict(test_X)))
# %%[markdown]
# That was amazing, so gradient boosting algorithm works amazing in this and gives an RMSE of 0.19
# This is extremely good, let's try a decision tree to see how that performs
# %%
# np.random.seed(7)
# model_dt=DecisionTreeRegressor(max_depth=9)
# model_dt.fit(train_X,train_Y)
# %%
# print('Train RMSE Decision Tree Regression: ',mean_squared_error(train_Y, model_dt.predict(train_X)))
# print('Test RMSE Decision Tree Regression: ',mean_squared_error(test_Y, model_dt.predict(test_X)))
# %%[markdown]
# We get a great model here as well, with an RMSE of 0.2. But the LGBM is still the better one.
# We'll consider LGBM as our best model. Now let's try to tune that for our final model
# 
# We'll use bayesian search instead of the regular grid search because our dataset (search space) is huge and
# we want a computationally efficient tuning algorithm
# %%
np.random.seed(7)
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31],
    "n_estimators": [100, 200],
    "max_depth": [-1, 8, 16],
    "reg_alpha": [1e-3, 1e-1]
}
bayesian_search = BayesSearchCV(model_lgbm, param_grid, n_iter=10, 
                                cv=3, scoring="neg_root_mean_squared_error")
np.int = int
bayesian_search.fit(train_X, train_Y)
# %%
print("Best parameters found: ", bayesian_search.best_params_)
# %%
print('Train RMSE LGBM Regression(Optimized): ',mean_squared_error(train_Y, bayesian_search.predict(train_X)))
print('Test RMSE LGBM Regression(Optimized): ',mean_squared_error(test_Y, bayesian_search.predict(test_X)))
# %%[markdown]
# Not much of an optimization but we have reduced the RMSE value from 0.1934 to 0.1919
# The best set of hyperparameters are:
# * Learning rate: 0.1
# * max_depth: 16
# * n_estimators: 153
# * num_leaves: 31
# * reg_alpha: 0.06383671801269114
# 
# The finale RMSE Value achieved:
# * Train RMSE: 0.1919
# * Test RMSE : 0.1924
# %%
# from sklearn.ensemble import RandomForestRegressor
# model_rf = RandomForestRegressor()
# model_rf.fit(train_X,train_Y)
# %%
# print('Train RMSE Random Forest Regression: ',mean_squared_error(train_Y, model_rf.predict(train_X)))
# print('Test RMSE Random Forest Regression: ',mean_squared_error(test_Y, model_rf.predict(test_X)))
# %%[markdown]
# We get an overfit model. This isn't worth the time and resources.
# %%
# from sklearn.ensemble import GradientBoostingRegressor
# model_gbr = GradientBoostingRegressor()
# model_gbr.fit(train_X,train_Y)
# %%
# print('Train RMSE Gradient Boosting Regression: ',mean_squared_error(train_Y, model_gbr.predict(train_X)))
# print('Test RMSE Gradient Boosting Regression: ',mean_squared_error(test_Y, model_gbr.predict(test_X)))
# %%[markdown]
# Basically the same result as decision tree but this consumes more time. So we'll exclude this.
# 
# From the bayesian search above, we have our best Parameters, let's build the LGBM Models with those params.
# %%
np.random.seed(7)
model_best_lgbm = LGBMRegressor(learning_rate=0.1, max_depth=16, n_estimators=153, 
                                num_leaves=31, reg_alpha=0.06383671801269114)
model_best_lgbm.fit(train_X, train_Y)
# %%
print('Train RMSE LGBM Regression: ',mean_squared_error(train_Y, model_best_lgbm.predict(train_X))) # 0.1918
print('Test RMSE LGBM Regression: ',mean_squared_error(test_Y, model_best_lgbm.predict(test_X))) # 0.1938
# %%
# Plot feature importance using Gain
plot_importance(model_best_lgbm, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
plt.show()
# %%
plot_importance(model_best_lgbm, importance_type="split", figsize=(7,6), title="LightGBM Feature Importance (Split)")
plt.show()
# %%[markdown]
# OK, So we have a general idea on which features seem to be imoprtant. 
# We need to remove the features which one can't provide input, luckily editor's choice is with the least importance,
# so we can easily remove it.
# %%
x = x.drop('Editors Choice', axis=1)
# %%
# Good, Now let's build the final model with the entire data and save it for production.
np.random.seed(7)
final_model = LGBMRegressor(learning_rate=0.1, max_depth=16, n_estimators=153, 
                                num_leaves=31, reg_alpha=0.06383671801269114)
final_model.fit(x, y)
# %%
mean_squared_error(y, final_model.predict(x))
# Not much of a drop, so we're good to go
# %%
pickle.dump(final_model, open('rating_model.pkl', 'wb'))
# %%
# model = pickle.load(open("rating_model.pkl", "rb"))
# # %%
# test1 = model.predict(x)
# %%
