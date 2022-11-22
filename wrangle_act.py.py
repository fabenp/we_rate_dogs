#!/usr/bin/env python
# coding: utf-8

# ## Data Gathering
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('twitter-archive-enhanced.csv')
df.head(2)


# In[3]:


df.retweeted_status_id.value_counts()


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.rating_denominator.value_counts()


# In[7]:


df.duplicated().sum()


# In[8]:


# source column is irrelevant and the doggo, floofer,puppo,pupper have only some data and are not relevant neither
df.source.unique() ,df.doggo.value_counts() ,df.floofer.value_counts() ,df.puppo.value_counts(), df.pupper.value_counts()


# 2. Use the Requests library to download the tweet image prediction (image_predictions.tsv)

# In[9]:


from bs4 import BeautifulSoup
import requests
url='https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)
with open ('image-predictions', mode ='wb') as file :
    file.write(response.content)
soup=BeautifulSoup(response.content,'lxml')


# In[10]:


new=pd.read_csv('image-predictions', sep='\t')
new.head(2)


# In[11]:


new.info()


# In[12]:


new.duplicated().sum()


# 3. Use the Tweepy library to query additional data via the Twitter API (tweet_json.txt)

# In[13]:


pip install tweepy


# In[14]:


import tweepy
from tweepy import OAuthHandler

#consumer_key = ###################
#consumer_secret = ###################
#access_token = ###################
#access_secret = ###################

#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_secret)

#api = tweepy.API(auth)


# In[15]:


tweet_ids = new.tweet_id.values


# In[16]:


#couldn't get tweet directly from tweeter, o downloaded the file form Udacity
import json
data = [json.loads(line)
        for line in open('tweet-json.txt', 'r', encoding='utf-8')]


# In[17]:


data=pd.DataFrame(data)


# In[18]:


data.columns


# In[19]:


data.head(2)


# In[20]:


data.extended_entities[0]


# In[21]:


data.user[0]


# ## Assessing Data

# ### Quality issues   
#   <font size=4>  1. Remove retweet from df table                 
# 
#    <font size=4>    2. Missing values in data and df tables
#        
#   <font size=4>  3. df timestamp is not in date format
#      
# 
#    <font size=4>    4. Data : tweet_id is id column : consistency issue between tables
# 
#   <font size=4>     5. Data table have irrelevant columns
# 
#   <font size=4>     6. df rating denominator column has values different from 10
#       
#  <font size=4>     7.  for the p1 columns, some images don't have a dog detected. the right breed detection can be in p2 , p3 or neither of the 3. A better way to visualize it is to replace the 6 column by one column for the breed detected
#       
#   <font size=4>     8.Each table should have one type of observational unit

# ### Tidiness issues
#   <font size=4> 1.df soucre,floofer,pupper,puppo and goddo columns should  e  in one column
# 
#   <font size=4>  2.Data table: entities, extended entities and user column have have many values

# ## Cleaning Data
# 

# In[22]:


# Make copies of original pieces of data
df1=df.copy()
data1=data.copy()
new1=new.copy()


# ### Issue #1: <font size =4 color= 'darkgreen'> remove retweets

# In[23]:


df1.head(2)


# In[24]:


df1=df1[df1.retweeted_status_id.isnull()]


# In[25]:


df.shape,df1.shape


# In[26]:


df1.columns


# ### Issue #2:
# 
#  <font size =4 color= 'darkgreen'> columns for retweets have  only missing values in df table.
#     
#   <font size =4 color= 'darkgreen'> columns starting with "in_reply_to" or "quoted_status", columnsn geo, coordinates, place,contributors have mainly missing values in data table

# In[27]:


df1.drop(['in_reply_to_status_id', 'in_reply_to_user_id','retweeted_status_id', 'retweeted_status_user_id','retweeted_status_timestamp'],axis=1, inplace=True)


# In[28]:


data1.drop(['geo','coordinates','place','contributors','retweeted_status',
       'quoted_status_id', 'quoted_status_id_str', 'quoted_status','lang','retweeted','in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name'],axis=1,inplace=True)


# In[29]:


data1.columns


# In[30]:


data.shape, data1.shape


# ### Issue #3:

# <font size =4 color= 'darkgreen'> df1 column timestamp is not in datetime format

# In[31]:


df1.timestamp=pd.to_datetime(df1.timestamp)


# In[32]:


df1.head(2)


# ### Issue #4:<font size =4 color= 'darkgreen'>  Column "id" has to be changes to 'tweet_id' to be consistent with other tables

# In[33]:


data1.rename(columns={'id':'tweet_id'}, inplace=True)


# In[34]:


data1.head(1)


# ### Issue #5:

# <font size =4 color= 'darkgreen'> data1 has many irrelvant columns : 
#     
# >id_str: same as  tweet_id
#     
# >truncated : all values are false
#     
# >display_text_range: a range  and not pertnent
#     
# >entities : same as extended_entities
#     
# 
# > user: the same information in all the rows
#     
# > possibly_sensitive and possibly_sensitive_appealable have only False as value
#     
# <font size =4 color= 'darkgreen'> df1 has many irrelvant columns :
#     
# > source column
#     
# > expnaded_urls : we have the URL in extened_entities
# 

# In[35]:


data1.head(2)


# In[36]:


data1.drop(['display_text_range','entities','id_str','truncated','user','possibly_sensitive','possibly_sensitive_appealable'
           ],axis=1,inplace=True)


# In[37]:


data1.head(1)


# In[38]:


df1.drop(['expanded_urls','source'],axis=1, inplace =True)


# In[39]:


df1.head(2)


# ## Tidiness issues

# ### Tidiness issue #1
# <font size =4 color= 'darkgreen'> The column extended_entities has a complicated structure of dictionary inide a list in side a dictionary. What we need to keep form this oclumn is the 'id' (not sure if important), the link to the tweet(display_url) and the link to the photo (media_url)

# In[40]:


data.extended_entities[1]


# In[41]:


data1['extended_entities'][0]


# In[42]:


data1['new_id']=1
data1['photo']=1
data1['link']=1


# In[43]:


index_na=data1.loc[data1['extended_entities'].isna()].index


# In[44]:


rows= (i for i in  range(0,2354) if i not in index_na )
for i in rows:
    
    if data1['extended_entities'][i] !='nan':
        data1['new_id'][i]=data1['extended_entities'][i]['media'][0]['id']
        data1['photo'][i]=data1['extended_entities'][i]['media'][0]['media_url']
        data1['link'][i]=data1['extended_entities'][i]['media'][0]['display_url']    


# In[45]:


data1.drop('extended_entities', axis=1, inplace=True)


# ### Tidiness issue #2 <font size =4 color= 'darkgreen'> df ,floofer,pupper,puppo and goddo columns should  be  in one column

# In[46]:


df1['sort'] = df1[['doggo','floofer','pupper','puppo']].apply(lambda x: '_'.join(x) , axis=1)


# In[47]:


df1['sort']=df1['sort'].apply( lambda x:x.replace('None',''))


# In[48]:


df1.sort.value_counts()


# In[49]:


df1.drop(['doggo','floofer','pupper','puppo'],axis=1, inplace=True)


# ### Issue #6: Fixed once tidiness issue fixed

# <font size =4 color= 'darkgreen'> Denominators with values > 10

# In[50]:


high=df1[df1.rating_denominator > 10]
high1=data1[data1.tweet_id.isin(high.tweet_id)]
high2=high1.loc[high1.photo != 1]


# In[51]:


from PIL import Image
import requests
from io import BytesIO

for i in high2.index:
    url = high2.photo[i]
    print(high2.full_text[i])
    response = requests.get(url,stream=True)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()


#   <font siez =4 color = 'darkgreen' >so the discrepancy in denominator is explained by a higher number of dogs in the same photo or in two cases by a typo. As we won't use the denominator columns, we can leave it as it is

# ### Issue #7
# <font size =4 color= 'darkgreen'>Dogs insome images were not detected, so there's no breed returned. However the right breed can be detected for  p2_dog or p3_dog

# In[52]:


no_dog=new1[new1.p1_dog ==False]
no_dog1= no_dog[(no_dog.p2_dog ==True) |(no_dog.p3_dog ==True)]
no_dog1


# In[53]:


for i in no_dog1.index:
    url = no_dog1.jpg_url[i]
    print( no_dog1.p1[i], no_dog1.p2[i],no_dog1.p3[i])
    response = requests.get(url,stream=True)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()


# In[54]:


new1.p1.value_counts()


# In[55]:


new1['breed']=1


# In[56]:


for i in new1.index:
    if new1.p1_dog[i] ==True:
        new1['breed'][i]=new1['p1'][i]
    elif new1.p2_dog[i]==True:
        new1['breed'][i]=new1['p2'][i]
    elif new1.p3_dog[i]==True:
        new1['breed'][i]=new1['p3'][i]
    else:
        new1['breed'][i]=False
            
            


# In[57]:


new1.columns


# In[58]:


new1.drop(['p1', 'p1_conf', 'p1_dog', 'p2','p2_conf', 'p2_dog', 'p3', 'p3_conf', 'p3_dog'], axis=1, inplace=True)


# ### Issue #7
# <font size =4 color= 'darkgreen'>Each table  should have one type of observational unit
#     Remove "_" form breed names

# In[59]:


new1.head()


# In[60]:


new1.breed=new1.breed.str.replace('_',' ')


# In[61]:


data1.drop(['photo','new_id','is_quote_status','favorited'],axis=1,inplace=True)


# In[62]:


data1.rename(columns={'created_at':'timestamp', 'link':'tweet_link','full_text':'tweet_text'},inplace=True)


# In[63]:


data1.head(1)


# In[64]:


data1.source.unique()


# In[65]:


df1.drop(['timestamp','text'],axis=1,inplace=True)


# In[66]:


df1.head(1)


# In[67]:


df1.name.unique()


# ## Storing Data

# In[68]:


new1.to_csv('newf.csv')
df1.to_csv('dff.csv')
data1.to_csv('data1f.csv')


# In[69]:


new1.head(1)


# In[70]:


df1.head(1)


# In[71]:


data1.head(1)


# In[72]:


dataset1=pd.merge(new1,df1, left_on='tweet_id',right_on='tweet_id',how='inner')
twitter_archive_master=pd.merge(dataset1,data1, left_on='tweet_id',right_on='tweet_id',how='inner')


# In[73]:


twitter_archive_master.to_csv('twitter_archive_master.csv')


# ## Analyzing and Visualizing Data
# In this section, analyze and visualize your wrangled data. You must produce at least **three (3) insights and one (1) visualization.**

# In[74]:


df1=pd.read_csv('newf.csv')
df2=pd.read_csv('dff.csv')
df3=pd.read_csv('data1f.csv')


# In[75]:


df3.head(1)


# In[76]:


pred=df1.copy()


# In[77]:


pred.drop('Unnamed: 0',axis=1,inplace=True)


# In[78]:


df2.head(2)


# In[79]:


dog_rate=df2.copy()


# In[80]:


#columns exist in the other tables
dog_rate.drop(['Unnamed: 0','rating_denominator'],axis=1,inplace=True)


# In[81]:


df3.head(1)


# In[82]:


count=df3.copy()


# In[83]:


dog_rate.head(2)


# In[84]:


count.head(2)


# In[85]:


pred.head(2)


# ### Visualization

# In[86]:


import matplotlib.pyplot as plt


# In[87]:


dog_rate.rating_numerator.value_counts()


# In[88]:


plt.hist(dog_rate.rating_numerator, bins=2000)
plt.xlim(7,14)
plt.xlabel('rating numerator')
plt.ylabel('count of rating')
plt.title('Rating counts for the most common numerator');


# In[89]:


breed= pd.merge(count,pred, left_on='tweet_id',right_on='tweet_id', how='inner')


# In[90]:


breed=breed.loc[breed.breed != 'False']


# In[91]:


breed


# In[92]:


breed_fav=breed[['favorite_count','breed']]


# In[93]:


breed_fav.groupby('breed')['favorite_count'].mean().sort_values(ascending=False)


# <font size=4>Saluki breed is only for 4 photos among which, one has a favourite count of 51000 which is mileading

# In[94]:


saluki=breed.loc[breed.breed=='Saluki']
saluki.shape


# In[95]:


# we will only visualize the top 10 breeds most popular
x=breed_fav.breed.value_counts()
breeds=x[0:10]
breed2=breed.loc[breed['breed'].isin(breeds.index)]


# In[96]:


plt.scatter( x=breed2['breed'],y=breed2.favorite_count)
plt.xticks(rotation=45)
ax = plt.gca()
x_labels=breed2['breed']
ax.set_xticklabels(x_labels, rotation=40,ha='right',va='top')
ax.set_xlabel('breed')
ax.set_ylabel('Favourite count')
ax.set_title('Top 10 favourited breeds')
plt.show()


# In[97]:


breed.breed.value_counts()


# A more accurate way to visualize the data is to focus on the breed with more photos

# In[98]:


# breed2 is limited to 10 most popular breeds (in terms of number of images)
breed2.to_csv('Breed_data.csv')


# In[99]:


import seaborn as sb

sb.boxplot(data=breed2, x='breed', y='favorite_count')
x_labels=breed2['breed'].unique()
ax = plt.gca()
ax.set_xticklabels(x_labels,ha='right',va='top')
plt.xticks(rotation=45)
ax.set_title('favourite count for the most common breeds in the dataset')
ax.set_xlabel('Breeds');


# # Check breed accuracy

# In[100]:


import urllib.request
from PIL import Image


# In[101]:


dfnew=df1[['jpg_url','breed']]
dfnew


# In[102]:


golden=dfnew.loc[dfnew.breed=='golden_retriever']
labrador=dfnew.loc[dfnew.breed=='Labrador_retriever']
pembroke=dfnew.loc[dfnew.breed=='Pembroke']
labrador=dfnew.loc[dfnew.breed=='Labrador_retriever']
chihuahua=dfnew.loc[dfnew.breed=='Chihuahua']
toy_poodle=dfnew.loc[dfnew.breed=='toy_poodle']


# In[103]:


from PIL import Image
import requests
from io import BytesIO

for i in labrador.index:
    url = labrador.jpg_url[i]
    response = requests.get(url,stream=True)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()


# In[104]:


from PIL import Image
import requests
from io import BytesIO

for i in toy_poodle.index:
    url = toy_poodle.jpg_url[i]
    response = requests.get(url,stream=True)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()


# For the Labrador breed, all the predictions are accurate
# For the top poodle, some photos have clearly another breed like yorshire or golden retriever, but stil the vast majority is accurately labeled
