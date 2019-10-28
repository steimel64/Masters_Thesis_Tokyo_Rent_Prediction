#!/usr/bin/env python
# coding: utf-8

# In[67]:


## Import Libraries 

## Shuffle 
import os
from sklearn.utils import shuffle

## Basic Data Processing
import pandas as pd
import numpy as np
pd.options.display.max_rows = 999
pd.options.display.max_columns = 50000

## Train/Test Split
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

##sparse
from scipy.sparse import vstack
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix

## Matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[70]:


## Change to Data Directory
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/raw')
os.getcwd()


# In[71]:


## Read Suumo Data 
df = pd.read_csv('tokyo.csv')
## remove unwanted characters
df.floor = df.floor.str.replace('階','').str.replace('\r\n\t\t\t\t\t\t\t\t\t\t\t','')


# In[72]:


# shuffle data
df = shuffle(df, random_state=13)


# In[73]:


## Get Total Samples 
totalsamples = df.shape[0]


# ## Create Target Value

# In[74]:


## Rent - Pre-Processing  
df.rent = df.rent.str.strip('万円').astype(float) * 10000

## Admin Fee
## - strip manen and add 0's 
df.admin.replace('-',0,inplace=True)
df.admin = df.admin.str.strip('円').astype(float)
## Fill NA's with 0 
df.admin.fillna(0,inplace=True)
## create admin flag
df['admin_flag'] = np.where(df['admin'] > 0,1,0)

## Create Target Variable (Rent + Admin Fee) 
df['target'] = df['rent'] + df['admin']


# In[75]:


## Drop Admin Fee - Its data leak 
df.drop(columns=['admin'], inplace=True)


# ## Outlier- Removal Noise

# In[76]:


## Drop Target above 800,000 Yen - Outliers Removal - Couldn't get distributions to match 
## Removed Rent below 23000 as apartments below this price are extreme outliers 
df = df[df['target'] < 800000]
df = df[df['target'] > 23000]


# In[77]:


print('Outliers Removed-', totalsamples - df.shape[0])
totalsamples = df.shape[0]


# #### Clean up Area / Yrs

# In[78]:


## Area - textcleanup- Remove meters squared 
df.area = df.area.str.replace('m2','')


# In[79]:


## Years text cleanup 
## Create new column for new building
df.yrs = df.yrs.str.strip('築').str.strip('年')
df['new'] = np.where(df.yrs == '新',1,0)
df.yrs.replace('新', '0',inplace=True)


# In[80]:


## Type Conversion 
df.area = df.area.astype('float')
df.yrs = df.yrs.astype('int')


# ## OUTLIER HANDLING - Rent Per M by Year

# In[81]:


## Calculate rent_per_m 
df['rent_per_m'] = df['target'] / df['area']


# In[82]:


## Drop 99 years - Looks like noise 
df = df[df['yrs'] != 99]


# In[83]:


## Yr Bins -- Create Bins for years above 50 as the samples dwindle 
df['yrbins'] = np.where((df['yrs'] >= 50) & (df['yrs'] < 60), '50s', df['yrs'])
df['yrbins'] = np.where((df['yrs'] >= 60) & (df['yrs'] < 70), '60s', df['yrs'])
df['yrbins'] = np.where((df['yrs'] >= 70) & (df['yrs'] < 80), '70s', df['yrbins'])
df['yrbins'] = np.where((df['yrs'] >= 80) & (df['yrs'] < 90), '80s', df['yrbins'])
df['yrbins'] = np.where((df['yrs'] >= 90) & (df['yrs'] < 100), '90s', df['yrbins'])


# In[84]:


## Detect Outliers 
outlierdetect = df.groupby(['Ku','yrbins'])['rent_per_m'].agg(['count','mean','min','max'])


# In[85]:


## Merge outlier detection to Ku 
df = df.merge(outlierdetect, how='left', left_on=['Ku','yrbins'], right_index=True)


# In[86]:


## Create flag for upper outlier - 
df['upper_outlier'] = np.where((df['mean'] * 5) < df['rent_per_m'], 1, 0)


# In[87]:


## Create flag for lower outlier 
df['lower_outlier'] = np.where(df['rent_per_m'] < (df['mean'] / 5), 1,0) 


# In[88]:


## lower_outlier - REMOVE 
df = df[df['lower_outlier'] != 1]

## upper_outlier - REMOVE 
df = df[df['upper_outlier'] != 1]


# In[89]:


## Column Cleanup- Remove garbage columns 
df.drop(columns=['rent_per_m','count','mean','min','max','upper_outlier','lower_outlier','yrbins'], inplace=True)


# In[90]:


print('Outliers Removed-', totalsamples - df.shape[0])
totalsamples = df.shape[0]
print('Total Samples after Outlier Removal', totalsamples)


# ## Outlier Handling - Room Layout

# In[91]:


## Deletion of Values with Ridiculous Room Layouts like 22LDK or 44k 
df = df[~df.floor_plan.isin(['22LDK','12K'])]


# In[92]:


print('Outliers Removed-', totalsamples - df.shape[0])
totalsamples = df.shape[0]


# ## Train/Test Split

# In[93]:


## Split Train/Test - 15 Bins 
df['target_bins'] = pd.qcut(df['target'],15)
train, test = train_test_split(df,stratify = df['target_bins'], test_size=.2, random_state=13)

## Split Train/Valid
train, valid = train_test_split(train, stratify = train['target_bins'], test_size=.2, random_state=13)

print('Price bins-', np.array(df['target_bins'].unique().sort_values()))

## Indexes 
train_in = train.index.values
valid_in = valid.index.values
test_in = test.index.values

## Combine Train and Test 
df = pd.concat([train, valid, test], axis=0)

## Create Column indicators 
df['train'] = np.where(df.index.isin(train_in),1,0)
df['valid'] = np.where(df.index.isin(valid_in),1,0)
df['test'] = np.where(df.index.isin(test_in),1,0)

df.drop(columns=['target_bins'], inplace=True)


# In[94]:


plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.5)


# In[95]:


## Plot Distribution for Train vs Validation Set 
fig, ax = plt.subplots(ncols=3, figsize=(14,5))
plt.subplot(1, 3, 1)
plt.title('Train vs Validation Set',weight='bold')
plt.hist(df[df['train'] == 1]['target'], bins = 25,rwidth=0.8, density=True,range=[df['target'].min(),df['target'].max()], alpha=.6, color='teal', label='Train', log=True)
plt.hist(df[df['valid'] == 1]['target'], bins = 25,rwidth=0.8,density=True,range=[df['target'].min(),df['target'].max()], alpha=.6, color='crimson', label='Valid', log=True)
plt.legend()

## Plot Distribution for Train vs Test Set 
plt.subplot(1, 3, 2)
plt.title('Train vs Validation Set',weight='bold')
plt.hist(df[df['train'] == 1]['target'], bins = 25,rwidth=0.8, density=True, range=[df['target'].min(),df['target'].max()], alpha=.6, color='teal', label='Train', log=True)
plt.hist(df[df['test'] == 1]['target'], bins = 25,rwidth=0.8,density=True, range=[df['target'].min(),df['target'].max()], stacked=True, alpha=.6, color='purple', label='Test', log=True) 
plt.legend()



## Plot Distribution for Train + Valid vs Test Set 
plt.subplot(1, 3, 3)
plt.title('Train & Valid vs Test Set',weight='bold')
plt.hist(df[(df['train'] == 1) | (df['valid'] == 1)]['target'], bins = 25,rwidth=0.8, density=True, range=[df['target'].min(),df['target'].max()], alpha=.6, color='orange', label='Train+Valid', log=True)
plt.hist(df[df['test'] == 1]['target'], bins = 25,rwidth=0.8,density=True, range=[df['target'].min(),df['target'].max()], stacked=True, alpha=.6, color='purple', label='Test', log=True)


plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('distributions.png')


# In[96]:


## Diagnostics Distribution of Target  
traindist = pd.DataFrame(train['target'].describe()).T.rename(index={'target':'train'})
validdist = pd.DataFrame(valid['target'].describe()).T.rename(index={'target':'valid'})
testdist = pd.DataFrame(test['target'].describe()).T.rename(index={'target':'test'})

## Print Diagnostics 
datasetstats = pd.concat([traindist,validdist,testdist])
datasetstats


# ## Pre-Processing

# In[97]:


## Log Transform Area- Saw this in another paper 
df['area'] = np.log(df['area'])


# In[98]:


## Gratuity - strip manen and add 0's 
df.gratuity = df.gratuity.str.strip('万円')
df.gratuity.replace('-',0, inplace=True)
df.gratuity = df.gratuity.astype(float) * 10000
## gratuity flag 
df.gratuity = np.where(df.gratuity >0,1,0)


# In[99]:


## Deposit - strip manen and add 0's 
df.deposit = df.deposit.str.strip('万円')
df.deposit.replace('-',0, inplace=True)
df.deposit = df.deposit.astype(float) * 10000
## Deposit Flag 
df.deposit = np.where(df.deposit >0, 1, 0)


# In[100]:


## Location 
df['location'] = df['location'].str.replace('東京都','')
## Imputations for inconsistent Data - Crime Data - Need to impute these to get links 
df['location'] = df['location'].str.replace("港区元赤坂２","港区元麻布２")
df['location'] = df['location'].str.replace("足立区西伊興町","足立区西伊興２")
df['location'] = df['location'].str.replace("江戸川区鹿骨町","江戸川区鹿骨２")
## Imputes for inconsistent Data - Disaster Data 
df['location'] = df['location'].str.replace("新宿区三栄町","新宿区四谷三栄町")


# In[101]:


## Floor Plan
df.floor_plan.replace('ワンルーム','1RM', inplace=True)


# In[102]:


## Height - Split To Two Columns - Above ground and under ground 
height = df.heights.str.split('地上',expand=True)

height[1] = np.where(height[1].isnull() == True, height[0],height[1])

height[0] = height[0].str.strip('地下')
height[0] = np.where(height[0].str.contains('階建'),0,height[0])
height[0] = np.where(height[0].str.contains('平屋'),0,height[0])

#平屋- 1 floor 
height[1] = height[1].str.strip('階建')
height[1] = height[1].replace('平屋','1')

height.columns = ['Basement_Depth','Height']


# In[103]:


## Add new height columns to DF 
df = pd.concat([df,height], axis=1).drop(columns='heights')


# In[104]:


## Floors 
## Entire House Feature 
df['entirehouse'] = np.where(df.floor == '-',1,0)

## Mezanine feature 
df['mezanine'] = np.where(df.floor.isin(['M2','M1','M3']),1,0)

## Multifloor Feature 
multifloors = ['2-1','1-3','9-10','4-5','3-1','3-2','2-3','8-9','3-4','B1-3','1-B1','B1-1','1-4','5-6','3-5','7-8','37-38','12-13','12-11','10-11','1-1','2-2','6-7'
              '2-5','B1-2','46-47','13-14','14-15','2-4','B2-1','2-B1','11-12']
  
df['multifloor'] = np.where(df.floor.isin(multifloors),1,0)

## Clean Floors - In the case of Multi floor- Highest Floor is Selected, Negative floor indicates highest floor is basement 
df['floor'] = df.floor.replace({'2-1': '2','1-3':'3','9-10':'10','4-5':'5','3-1':'3','3-2':'3','2-3':'3','8-9':'9','3-4':'4','B1-3':'-1','1-B1':'1','B1-1':'1','1-4':'4','5-6':'6','3-5':'5','7-8':'8','37-38':'38',
          '12-13':'13','12-11':'12','10-11':'11','1-1':'1','2-2':'2','6-7':'7','2-5':'5','B1-2':'-1','46-47':'47','13-14':'14','14-15':'15','2-4':'4','B2-1':'-1','2-B1':'2','11-12':'12',
                               '1-2':'2','B1':'-1','B2':'-2', 'M2':'2','M1':'1', 'M3':'3', 'B14':'14', 'B10':'10', 'B3':'3', 'B8':'8', 'B13':'13', 'B4':'4', 'B':'-1', 'B5':'5'})

## - Value usually indicates entire house, height of building is used 
df.floor = np.where(df.floor == '-',df.Height, df.floor)

## Replace floor typos -B higher floors were encoded as positive because possible typo - Using building height and basement height as proxy 
df['floor'] = np.where(df['floor'].astype(int) <df['Basement_Depth'].astype(int)*-1, df['floor'].astype(int)*-1,df['floor'])

## Lives in basement 
basementdwellers = ['-2','-1']
df['basementdweller'] = np.where(df.floor.isin(basementdwellers),1,0)


# In[105]:


## Type Conversion
df.area = df.area.astype('float').round(2)
df.Basement_Depth = df.Basement_Depth.astype('int')
df.Height = df.Height.astype('int')
df.yrs = df.yrs.astype('int')
df.rent = df.rent.astype('int32')
df.target = df.target.astype('int32')


# ## Text Processing

# In[106]:


## Station 1 
split = df['station1'].str.split(' ',expand=True)
## Drop Column 2 its garbage
split.drop(columns=2,inplace=True)
## Car
split['Car_1'] = np.where(split[1].str.contains('車'),split[1],None)
## Walk 
split['Walk_1'] = np.where(split[3].isnull()==True,split[1],split[3])
split['Walk_1'] = np.where(split['Walk_1'].str.contains('車') == True,None, split['Walk_1'])
## Can drop 3 now that we have extracted all the information 
split.drop(columns=3, inplace=True)

##Bus 
split['Bus_1'] = np.where(split[1].str.contains('バス'),split[1],None)
split.drop(columns=1, inplace=True)

## Remove Kanji/Kana from Car, Walk, Bus 
split['Walk_1'] = split['Walk_1'].str.replace('歩','').str.replace('分','')
split['Car_1'] = split['Car_1'].str.replace('車','').str.split('分',expand=True)[0]
split['Bus_1'] = split['Bus_1'].str.replace('バス','').str.replace('分','')



## Split Station and Line 
split2 = split[0].str.split('/', expand=True)
split2.columns = ['Line_1','Station_1']
split2

station1info = pd.concat([split,split2],axis=1).drop(columns=0)

## Impute 1 incorrect value 
station1info['Bus_1'][station1info.index==30223] = '10'
station1info['Station_1'][station1info.index==30223] = '葛西駅'
station1info['Line_1'][station1info.index==30223] = station1info['Line_1'][station1info.index==30223].str.replace('葛西駅','')


# In[107]:


## Station 2 
## Ratio between two stations 

split = df['station2'].str.split(' ',expand=True)
## Drop Column 2 its garbage
split.drop(columns=2,inplace=True)
## Car
split['Car_2'] = np.where(split[1].str.contains('車'),split[1],None)
## Walk 
split['Walk_2'] = np.where(split[3].isnull()==True,split[1],split[3])
split['Walk_2'] = np.where(split['Walk_2'].str.contains('車') == True,None, split['Walk_2'])
## Can drop 3 now that we have extracted all the information 
split.drop(columns=3, inplace=True)


##Bus 
split['Bus_2'] = np.where(split[1].str.contains('バス'),split[1],None)
split.drop(columns=1, inplace=True)

## Little Cleaning 
split['Walk_2'][split.index == 3082] = '歩3分'
split.drop(columns=4, inplace=True)

## Remove Kanji/Kana from Car, Walk, Bus 
split['Walk_2'] = split['Walk_2'].str.replace('歩','').str.replace('分','')
split['Car_2'] = split['Car_2'].str.replace('車','').str.split('分',expand=True)[0]
split['Bus_2'] = split['Bus_2'].str.replace('バス','').str.replace('分','')

## Split Station and Line 
split2 = split[0].str.split('/', expand=True)
split2.columns = ['Line_2','Station_2']

station2info = pd.concat([split,split2],axis=1).drop(columns=0)

## Replace NaN Values with None- Had to create a workaround as could not easily convert NaN to None 
station2info['Walk_2'] = station2info['Walk_2'].replace(np.nan, 'NaN')
station2info['Walk_2'] = np.where(station2info['Walk_2'] == 'NaN',None,station2info['Walk_2'])


station2info['Car_2'] = station2info['Car_2'].replace(np.nan, 'NaN')
station2info['Car_2'] = np.where(station2info['Car_2'] == 'NaN',None,station2info['Car_2'])


station2info['Bus_2'] = station2info['Bus_2'].replace(np.nan, 'NaN')
station2info['Bus_2'] = np.where(station2info['Bus_2'] == 'NaN',None,station2info['Bus_2'])


station2info['Line_2'] = station2info['Line_2'].replace(np.nan, 'NaN')
station2info['Line_2'] = np.where(station2info['Line_2'] == 'NaN','No_Line_2',station2info['Line_2'])

station2info['Station_2'] = station2info['Station_2'].replace(np.nan, 'NaN')
station2info['Station_2'] = np.where(station2info['Station_2'] == 'NaN',"No_Station_2",station2info['Station_2'])


# In[108]:


## Station 3 
split = df['station3'].str.split(' ',expand=True)
## Drop Column 2 its garbage
split.drop(columns=2,inplace=True)
## Car
split['Car_3'] = np.where(split[1].str.contains('車'),split[1],None)
## Walk 
split['Walk_3'] = np.where(split[3].isnull()==True,split[1],split[3])
split['Walk_3'] = np.where(split['Walk_3'].str.contains('車') == True,None, split['Walk_3'])


## Can drop 3 now that we have extracted all the information 
split.drop(columns=3, inplace=True)


##Bus 
split['Bus_3'] = np.where(split[1].str.contains('バス'),split[1],None)
split.drop(columns=1, inplace=True)

## Remove Kanji/Kana from Car, Walk, Bus 
split['Walk_3'] = split['Walk_3'].str.replace('歩','').str.replace('分','')
split['Car_3'] = split['Car_3'].str.replace('車','').str.split('分',expand=True)[0]
split['Bus_3'] = split['Bus_3'].str.replace('バス','').str.replace('分','')

## Split Station and Line 
split2 = split[0].str.split('/', expand=True)
split2.columns = ['Line_3','Station_3']

station3info = pd.concat([split,split2],axis=1).drop(columns=0)
station3info.head()

## Replace NaN Values with None- Had to create a workaround as could not easily convert NaN to None 
station3info['Walk_3'] = station3info['Walk_3'].replace(np.nan, 'NaN')
station3info['Walk_3'] = np.where(station3info['Walk_3'] == 'NaN',None,station3info['Walk_3'])


station3info['Car_3'] = station3info['Car_3'].replace(np.nan, 'NaN')
station3info['Car_3'] = np.where(station3info['Car_3'] == 'NaN',None,station3info['Car_3'])


station3info['Bus_3'] = station3info['Bus_3'].replace(np.nan, 'NaN')
station3info['Bus_3'] = np.where(station3info['Bus_3'] == 'NaN',None,station3info['Bus_3'])


station3info['Line_3'] = station3info['Line_3'].replace(np.nan, 'NaN')
station3info['Line_3'] = np.where(station3info['Line_3'] == 'NaN','No_Line_3',station3info['Line_3'])

station3info['Station_3'] = station3info['Station_3'].replace(np.nan, 'NaN')
station3info['Station_3'] = np.where(station3info['Station_3'] == 'NaN','No_Station_3',station3info['Station_3'])

## Clean 4 Incorrect Values

## 43148
## Set Station to 葛飾車庫入口 
## Set Line to 有27 亀有駅北口行
## Set Walk_3 to 4 

station3info['Station_3'][station3info.index == 43148] = '葛飾車庫入口'
station3info['Line_3'][station3info.index == 43148] = '有27 亀有駅北口行'
station3info['Walk_3'][station3info.index == 43148] = 4
station3info['Car_3'][station3info.index == 43148] = None

## 42011 
## Set Station to 金町駅 
## Set Line to 小55 小岩駅行
## Set Walk_3 to 16 


station3info['Station_3'][station3info.index == 42011] = '金町駅'
station3info['Line_3'][station3info.index == 42011] = '小55 小岩駅行'
station3info['Walk_3'][station3info.index == 42011] = 16

## 64276
## Set Station to 円融寺前
## Set Line to 渋谷～洗足間
## Set Walk_3 to 5 
station3info['Station_3'][station3info.index == 64276] = '円融寺前'
station3info['Line_3'][station3info.index == 64276] = '渋谷～洗足間'
station3info['Walk_3'][station3info.index == 64276] = 5



## 100731 
## Set Station to 大森西４丁目
## Set Line to 森50 東邦大学
## Walk to 1 

station3info['Station_3'][station3info.index == 100731] = '大森西４丁目'
station3info['Line_3'][station3info.index == 100731] = '森50 東邦大学'
station3info['Walk_3'][station3info.index == 100731] = 1


# In[109]:


## Final Combine 
## station2info - add to concat
## station3info - add to concat

finaldf = pd.concat([df,station1info,station2info,station3info],axis=1)
finaldf.drop(['station1','station2','station3'],axis=1,inplace=True)


# In[110]:


## Update D types 
finaldf['Car_1']= finaldf['Car_1'].astype(float)
finaldf['Bus_1']= finaldf['Bus_1'].astype(float)
finaldf['Walk_1']= finaldf['Walk_1'].astype(float)
finaldf['Car_2']= finaldf['Car_2'].astype(float)
finaldf['Bus_2']= finaldf['Bus_2'].astype(float)
finaldf['Walk_2']= finaldf['Walk_2'].astype(float)
finaldf['Car_3']= finaldf['Car_3'].astype(float)
finaldf['Bus_3']= finaldf['Bus_3'].astype(float)
finaldf['Walk_3']= finaldf['Walk_3'].astype(float)
finaldf['floor'] = finaldf['floor'].astype('int32')


# In[111]:


def reduce_mem_usage(props,prompt=True):
    nan_cols=props.columns[props.isnull().any()].tolist()
    if prompt:
        start_mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            if prompt:
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",props[col].dtype)
            if col in nan_cols:
                if prompt: 
                    print('Column: %s has NAN values'%col)
                props.loc[:,col] = props.loc[:,col].astype(np.float32)
            else:
                # make variables for Int, max and min
                IsInt = False
                mx = props[col].max()
                mn = props[col].min()
                
                # Integer does not support NA, therefore, NA needs to be filled

                # test if column can be converted to an integer
                asint = props[col].astype(np.int64)
                result = (props[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True
                
                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 2**8:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint8)
                        elif mx < 2**16:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint16)
                        elif mx < 2**32:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint32)
                        else:
                            props.loc[:,col] = props.loc[:,col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props.loc[:,col] = props.loc[:,col].astype(np.int64)	

                # Make float datatypes 32 bit
                else:
                    props.loc[:,col] = props.loc[:,col].astype(np.float32)

            if prompt:
                # Print new column type
                print("dtype after: ",props[col].dtype)
                print("******************************")

    if prompt:
       # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props


# ## Create Dataset - Baseline Model

# In[112]:


## First a function for pre_processing automation is defined 
def pre_process(df, dropcolumns): 
    df = reduce_mem_usage(df)
  
    train = df[df['train'] == 1]
    valid = df[df['valid'] == 1]
    test = df[df['test'] == 1]
  
    ## n_train shape, n_valid shape, n_test shape 
    n_train = train.shape[0]
    n_valid = valid.shape[0]
    n_test = test.shape[0]
  
    ## Split Train, Valid, Test Set 
    y_train = train.target
    y_valid = valid.target
    y_test = test.target
    X_train = train.drop(columns='target')
    X_valid = valid.drop(columns='target')
    X_test = test.drop(columns='target')
  
    ## Drop Unneed Columns / Split Dataset 
    X_train.drop(columns=dropcolumns, axis=1, inplace=True)
    X_valid.drop(columns=dropcolumns, axis=1, inplace=True)
    X_test.drop(columns=dropcolumns, axis=1, inplace=True) 
  
  
    ## Get Distributions 
    ## Diagnostics Distribution of Target  
    traindist = pd.DataFrame(y_train.describe()).T.rename(index={'target':'train'})
    validdist = pd.DataFrame(y_valid.describe()).T.rename(index={'target':'valid'})
    testdist = pd.DataFrame(y_test.describe()).T.rename(index={'target':'test'})
  
    datasetstats = pd.concat([traindist,validdist,testdist])
    print('------Data Distribution------')
    print(datasetstats)
  

    ## Scaling Steps- NUMERIC FEATURES 
    numerics = ['uint8','uint16','uint32','int8','int16', 'int32', 'int64','float16', 'float32', 'float64']
    print(X_train.shape, X_valid.shape, X_test.shape)
    print('-----------------------')
    sc = StandardScaler()
    X_train_num = sc.fit_transform(X_train.drop(['train','valid','test'],axis=1).select_dtypes(include=numerics))
    X_valid_num = sc.transform(X_valid.drop(['train','valid','test'],axis=1).select_dtypes(include=numerics))
    X_test_num = sc.transform(X_test.drop(['train','valid','test'],axis=1).select_dtypes(include=numerics))
  
    numeric_column_names = X_train.drop(['train','valid','test'],axis=1).select_dtypes(include=numerics).columns
    print('Number of Numeric Columns-', numeric_column_names.shape[0])
    print('Numeric Columns-',numeric_column_names.values)
    print('-----------------------')
  
    ## Stack Numeric Data 
    full_num_data = vstack([X_train_num,X_valid_num, X_test_num])
  
    ## Combine Full Data 
    full_data = pd.concat([X_train,X_valid, X_test], axis=0)
  
    ## Collect Object Columns 
    objectcolumns = full_data.select_dtypes(['object','category']).columns
    print('Categorical Columns=', objectcolumns.values, 'shape=',objectcolumns.shape[0])
  
    ## Loop LabelBinarizer
    full_object_data = csr_matrix((0,0))
    object_column_names = np.array([])
    ## Binarizer 
    lb = LabelBinarizer()
    for i in objectcolumns: 
        labels = lb.fit_transform(full_data[i])
        columns = lb.classes_
        print(i,'features-',labels.shape[1])
        full_object_data = sparse.hstack([full_object_data,labels])
        object_column_names = np.concatenate((object_column_names, columns))
    
  ## Stack Numeric / Object Data 
    X = hstack((full_object_data, full_num_data)).tocsr()
    allcolumns = np.concatenate((object_column_names, numeric_column_names))
    print('Categorical Features Total=', X.shape[1] -numeric_column_names.shape[0])
    print('-----------------------')
    print('Full Shape =', X.shape)
  
  ## Create Final X_train, X_valid, and X_test
    X_train = X[:n_train]
    X_valid = X[n_train:n_train+n_valid]
    X_test = X[n_train+n_valid:]
    print('----------------------')
  
    print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)
  
    return(X_train,X_valid,X_test, allcolumns,y_train,y_valid,y_test)


# In[113]:


X_train, X_valid, X_test, allcolumns, y_train, y_valid, y_test = pre_process(finaldf,['title','rent'])


# In[114]:


## Save Training, Valid, and Test Files 

os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/train')

sparse.save_npz('X_train_baseline.npz', X_train)
y_train.to_hdf('y_train_baseline.hdf', 'train')

os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/valid')
sparse.save_npz('X_valid_baseline.npz', X_valid)
y_valid.to_hdf('y_valid_baseline.hdf', 'valid')

os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/test')
sparse.save_npz('X_test_baseline.npz', X_test)
y_test.to_hdf('y_test_baseline.hdf','test')


## Columns for LGBM 
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/')
pd.DataFrame(allcolumns).to_csv('columns_baseline.csv',index=False)

#pd.DataFrame(finaldf).to_csv('baseline_EDA.csv',index=False)


# ## GeoLocation Data (Index Reset)

# In[115]:


os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/raw')


# In[116]:


## Import Longitude Latitudes/ Location Data 
locationsref = pd.read_csv('locationspp.csv')
ekiref = pd.read_csv('stationspp.csv')


# In[117]:


## Drop Tokyo 
locationsref['location'] = locationsref.location.str.replace("東京都","")


# In[118]:


## Merge Geo Data to Locations 
finaldf = finaldf.merge(locationsref.set_index("location"), left_on="location", right_index = True)


# In[119]:


## Merge Geo Data to Locations 
finaldf = finaldf.merge(ekiref.set_index("Stations"), left_on="Station_1", right_index=True,suffixes=('', '_Eki1'))
finaldf = finaldf.merge(ekiref.set_index("Stations"), how='left',left_on="Station_2", right_index=True,suffixes=('', '_Eki2'))
finaldf = finaldf.merge(ekiref.set_index("Stations"), how='left',left_on="Station_3", right_index=True,suffixes=('', '_Eki3'))


# In[120]:


## Station 1 Flags - Add Car_1, Walk_1, Bus_1 - create flags 
finaldf['Time_to_Station1'] = finaldf.fillna(0)['Car_1'] + finaldf.fillna(0)['Walk_1'] + finaldf.fillna(0)['Bus_1']
finaldf['WalkFlag'] = np.where(finaldf['Walk_1']>0,1,0)
finaldf['BusFlag'] = np.where(finaldf['Bus_1']>0,1,0)
finaldf['CarFlag'] = np.where(finaldf['Car_1']>0,1,0)


# In[121]:


## Station 2 Flags - Add Car_2, Walk_2, Bus_2 - create flags 
finaldf['Time_to_Station2'] = finaldf.fillna(0)['Car_2'] + finaldf.fillna(0)['Walk_2'] + finaldf.fillna(0)['Bus_2']
finaldf['Time_to_Station2'] = np.where(finaldf['Time_to_Station2'] == 0, None,finaldf['Time_to_Station2']).astype('float32')
finaldf['WalkFlag_2'] = np.where(finaldf['Walk_2']>0,1,0)
finaldf['BusFlag_2'] = np.where(finaldf['Bus_2']>0,1,0)
finaldf['CarFlag_2'] = np.where(finaldf['Car_2']>0,1,0)


# In[122]:


## Station 3 Flags - Add Car_3, Walk_3, Bus_3 - create flags 
finaldf['Time_to_Station3'] = finaldf.fillna(0)['Car_3'] + finaldf.fillna(0)['Walk_3'] + finaldf.fillna(0)['Bus_3']
finaldf['Time_to_Station3'] = np.where(finaldf['Time_to_Station3'] == 0, None,finaldf['Time_to_Station3']).astype('float32')
finaldf['WalkFlag_3'] = np.where(finaldf['Walk_3']>0,1,0)
finaldf['BusFlag_3'] = np.where(finaldf['Bus_3']>0,1,0)
finaldf['CarFlag_3'] = np.where(finaldf['Car_3']>0,1,0)


# In[123]:


## Read Distance Matrixes 
distmatrix = pd.read_csv('Dist_Matrix.csv')
distmatrix.drop(columns='Unnamed: 0', inplace=True)


# In[124]:


## Link Distances to Station1 
finaldf = finaldf.merge(distmatrix.set_index("Stations"), left_on="Station_1", right_index = True)


# In[125]:


## Rename Columns again 
finaldf.rename(columns={'LAT_x':'LAT','LON_x':'LON'}, inplace=True)


# In[126]:


## Drop Unneeded columns 
finaldf.drop(columns=['Car_1','Walk_1','Bus_1','Eng_Location','LAT_y','LON_y','Lat_Lon','Eng_Location_Eki1','Car_2','Walk_2','Bus_2','Eng_Location_Eki2','Car_3','Walk_3','Bus_3','Eng_Location_Eki3'], inplace=True)


# In[127]:


## Clean up distance columns 
finaldf.ShinjukuDist = finaldf.ShinjukuDist.str.replace('km','').str.strip()
finaldf.TokyoDist = finaldf.TokyoDist.str.replace('km','').str.strip()
finaldf.ShibuyaDist = finaldf.ShibuyaDist.str.replace('km','').str.strip()
finaldf.IkebukuroDist = finaldf.IkebukuroDist.str.replace('km','').str.strip()
finaldf.UenoDist = finaldf.UenoDist.str.replace('km','').str.strip()
finaldf.ShinagawaDist = finaldf.ShinagawaDist.str.replace('km','').str.strip()
finaldf.ShinjukuDist = finaldf['ShinjukuDist'].astype('float32')
finaldf.TokyoDist = finaldf['TokyoDist'].astype('float32')
finaldf.ShibuyaDist = finaldf['ShibuyaDist'].astype('float32')
finaldf.IkebukuroDist = finaldf['IkebukuroDist'].astype('float32')
finaldf.UenoDist = finaldf['UenoDist'].astype('float32')
finaldf.ShinagawaDist = finaldf['ShinagawaDist'].astype('float32')


# ## Crime

# In[129]:


## Read Tokyo Crime data 
crime = pd.read_csv('Crime.csv')


# In[130]:


## Clean crime data so it links with the original data
crime['Location'] = crime['Location'].str.replace(' ','')
crime['Location'] = crime['Location'].str.replace('丁目','')


# In[131]:


## Impute Crime Data that is not available with 0's 
## 台東区谷中４ Impute 0 
## 北区堀船４ Impute 0 
## 新宿区市谷砂土原町３ Impute 0 
## 新宿区市谷左内町 Impute 0 
## 新宿区二十騎町 Impute 0 
## 新宿区市谷甲良町 Impute 0 
## 千代田区神田東紺屋町 Impute 0 
## 千代田区西神田３ Impute 0 
## 杉並区成田西４ Impute 0 
## 港区麻布狸穴町 Impute 0 
## 港区麻布永坂町 Impute 0 

imputedf = pd.DataFrame(columns= crime.columns)
imputedf['Location'] = np.array(['新宿区二十騎町', '千代田区西神田３', '新宿区市谷砂土原町３', '台東区谷中４', '杉並区成田西４', '北区堀船４',
       '千代田区神田東紺屋町', '新宿区市谷甲良町', '港区麻布狸穴町', '港区麻布永坂町', '新宿区市谷左内町'])
imputeddf = imputedf.fillna(0)
crime = pd.concat([crime,imputeddf], ignore_index=True)


# In[132]:


## Merge Crime to base df 
finaldf = finaldf.merge(crime.set_index("Location"), left_on="location", right_index=True)


# In[133]:


## Rename Total to Total Crimes for ease 
finaldf.rename(columns={'Total':'Total_Crimes'}, inplace=True)


# ## Environmental Data

# ### Air Quality

# In[134]:


## Read Air Quality Data 
airquality = pd.read_csv('Air_Quality.csv')


# In[135]:


## Clean Data so that it can be merged with main dataframe 
airquality['location'] = airquality.location.str.replace("東京都","")


# In[136]:


## Merge by Location 
finaldf = finaldf.merge(airquality.set_index("location"), left_on="location", right_index = True)


# In[137]:


## Drop Unnecessary columns and rename 
finaldf.drop(columns=['Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','LAT_y','LON_y','Eng_Location'], inplace=True)
finaldf.rename(columns={'LAT_x':'LAT','LON_x':'LON'}, inplace=True)


# ### Bousai

# In[138]:


## Read Earthquake Data 
bousai = pd.read_csv('Bousai.csv')


# In[139]:


## Combine 
bousai['location'] = (bousai['Ku'].map(str) + bousai['location'].map(str))

## Replace Strings so that it can be joined 
bousai['location'] = bousai['location'].str.replace("　","")
bousai['location'] = bousai['location'].str.replace("丁目","")


# In[140]:


## Fix mismatches so that it can be joined 
## 千代田区猿楽町１ >> 千代田区神田猿楽町２
## 千代田区猿楽町２ >> 千代田区神田猿楽町２
## 千代田区三崎町１ >> 千代田区神田三崎町１
## 千代田区三崎町２ >> 千代田区神田三崎町２
## 千代田区三崎町３ >> 千代田区神田三崎町３
## 新宿区本塩町 >> 新宿区四谷本塩町
## 新宿区三栄町 >> 新宿区四谷三栄町

bousai['location'] = bousai['location'].str.replace("千代田区猿楽町１","千代田区神田猿楽町１")
bousai['location'] = bousai['location'].str.replace("千代田区猿楽町２","千代田区神田猿楽町２")
bousai['location'] = bousai['location'].str.replace("千代田区三崎町１","千代田区神田三崎町１")
bousai['location'] = bousai['location'].str.replace("千代田区三崎町２","千代田区神田三崎町２")
bousai['location'] = bousai['location'].str.replace("千代田区三崎町３","千代田区神田三崎町３")
bousai['location'] = bousai['location'].str.replace("新宿区本塩町","新宿区四谷本塩町")
bousai['location'] = bousai['location'].str.replace("新宿区三栄町","新宿区四谷三栄町")


# In[141]:


## Merge Data 
finaldf = finaldf.merge(bousai.set_index("location"), left_on="location", right_index = True)


# In[142]:


## Rename Columns 
finaldf.rename(columns={'Ku_x':'Ku'}, inplace=True)
finaldf.drop(columns=['Ku_y'], inplace=True)


# ## Parks

# In[144]:


## Read Park Data 
parks = pd.read_csv('parkdata.csv')


# In[145]:


## Merge Park Data to main df 
finaldf = finaldf.merge(parks.set_index("Ku"), left_on="Ku", right_index = True)


# In[146]:


## Rename and drop columns 
finaldf.rename(columns={'Area':'Parks_Area','Total':'Total_Parks'}, inplace=True)
finaldf.drop(columns=['Ku_japanese'], inplace=True)


# ## Land Use 

# In[147]:


## Merge Land Use by District Data 
landusedistrict = pd.read_csv('LandUsebyDistrict.csv').drop(columns='Unnamed: 14')


# In[148]:


## Drop Japanese Ku again 
landusedistrict.drop(columns='JapaneseKu', inplace=True)


# In[149]:


## Merge to main DF 
finaldf = finaldf.merge(landusedistrict.set_index("Ku"),left_on="Ku", how='left',right_index=True)


# In[150]:


## Rename ambigious columns 
landusedistrict.rename(columns={'Total':'Total_LandUse'}, inplace=True)


# In[151]:


## Rename columns in DF 
finaldf.rename(columns={'Total':'Total_District_Land'},inplace=True)


# ## Pollution Complaints

# In[152]:


## Read Pollutions complaints data 
pollutioncomplaints = pd.read_csv('PollutionComplaints.csv').drop(index=[23,24,25,26,27,28,29], columns=['Unnamed: 11',	'Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17',
                                                                                   'Unnamed: 18', 'Unnamed: 19','Unnamed: 20',	'Unnamed: 21',	'Unnamed: 22',	'Unnamed: 23'])


# In[153]:


## Drop Japanese KU 
pollutioncomplaints.drop(columns=['Japanese Ku'], inplace=True)


# In[154]:


## Replace the dashes with 0's 
pollutioncomplaints['Water pollution'] = pollutioncomplaints['Water pollution'].replace('-',0)
pollutioncomplaints['Soil pollution'] = pollutioncomplaints['Soil pollution'].replace('-',0)
pollutioncomplaints['ground_subsidence'] = pollutioncomplaints['ground_subsidence'].replace('-',0)
pollutioncomplaints['other_pollution'] = pollutioncomplaints['other_pollution'].replace('-',0)


# In[155]:


## Merge Pollution Complains 
finaldf = finaldf.merge(pollutioncomplaints.set_index("Ku"),left_on="Ku", how='left',right_index=True)


# In[156]:


## Drop Completely Empty features
finaldf.drop(columns=['Violent_Crime_Weapons','NO(ppm)_min','SPM(mg/m3)_min','WS(m/s)_min','ground_subsidence'],inplace=True)


# In[157]:


## fix dtypes
## Fix Dtypes 
finaldf['Population'] = finaldf['Population'].str.replace(',','').astype('int32')
finaldf['Population Density'] = finaldf['Population Density'].str.replace(',','').astype('float32')
finaldf['UP_Metropolitan_Parks_Area'] = finaldf['UP_Metropolitan_Parks_Area'].str.replace(',','').astype('float32')
finaldf['UP_Municipal_Parks_Area'] = finaldf['UP_Municipal_Parks_Area'].str.replace(',','').astype('float32')
finaldf['UP_National_Government_Parks_Area'] = finaldf['UP_National_Government_Parks_Area'].str.replace(',','').astype('float32')
finaldf['UP_Area'] = finaldf['UP_Area'].str.replace(',','').astype('float32')
finaldf['Total_District_Land'] = finaldf['Total_District_Land'].str.replace(',','').astype('float32')
finaldf['Residentail'] = finaldf['Residentail'].str.replace(',','').astype('float32')
finaldf['Roads'] = finaldf['Roads'].str.replace(',','').astype('float32')
finaldf['roads2'] = finaldf['roads2'].str.replace(',','').astype('float32')
finaldf['Water pollution'] = finaldf['Water pollution'].astype('float32')
finaldf['Soil pollution'] = finaldf['Soil pollution'].astype('float32')
finaldf['other_pollution'] = finaldf['other_pollution'].astype('float32')


# ## Create Dataset GeoSpatial/Environmental Features Model

# In[158]:


X_train, X_valid, X_test, allcolumns, y_train, y_valid, y_test = pre_process(finaldf,['title','rent'])


# In[197]:


## Save Training, Valid, and Test Files 
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/train')

sparse.save_npz('X_train_geo_env.npz', X_train)
y_train.to_hdf('y_train_geo_env.hdf', 'train')

os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/valid')
sparse.save_npz('X_valid_geo_env.npz', X_valid)
y_valid.to_hdf('y_valid_geo_env.hdf', 'valid')

os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/test')
sparse.save_npz('X_test_geo_env.npz', X_test)
y_test.to_hdf('y_test_geo_env.hdf','test')


## Columns for LGBM 
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/')
pd.DataFrame(allcolumns).to_csv('columns_geo_env.csv',index=False)


# ## Feature Engineering

# Constructed Feature Types 
# * Expansion Encoding 
#  * floorplan expanded to L, D, K, S, RM, and total_rooms (6)
# * Interaction Features 
# ##### BUILDING RELATED 
#  * area per room (area/rms) 
#  * height ratio - floor / height 
#  * area_height_int - area * height 
#  * floor_area_int - floor * area
# ##### Disaster Related 
#  * yrs_risk_interact - yrs * Total_Risk 
#  * yrs_lifediff_interact - yrs * lifediff 
#  * yrs_fire_interact - yrs * total fire risk 
#  * yrs_building_interact - yrs * building structure risk 
#  * pre1981 - building built pre 1981 
#  * pre1981_risk_interact - pre1981 * total_risk 
#  * pre1981_lifediff_interact - pre1981 * lifediff 
#  * pre1981_fire_interact - pre1981 * Fire risk 
#  * pre1981_building_interact - pre 1981 * building structure risk 
#  * pre1950 - building built pre 1950 
#   
# ##### COMMUTE 
#  * Hub_aggregate - Sum of all Time from station 1 to Hub Stations 
#  * Shinjuku_Commute - Walk to station1 + Commute to Shinjuku 
#  * Tokyo_Commute - Walk to station1 + Commute to Tokyo
#  * Shibuya_Commute  - Walk to station1 + Commute to Shibuya
#  * Ikebukuro_Commute - Walk to station1 + Commute to Ikebukuro 
#  * Shingawa_Commute - Walk to station1 + Commute to Shinagawa
#  * Ueno_Commute - Walk to station1 + Commute to Ueno 
#  * Aggregate_Commute - Sum of all Commutes
# ##### COMMUTE INTERACTIONS 
#  * agg_commute_floor_area_int = Aggregate_Commute * floor_area_int
#  * agg_commute_area_height_int = Aggregate_Commute * area_height_int
#  * agg_commute_area_per_room_int= Aggregate_Commute * area_per_room
#  * agg_commute_yrs_risk_int = Aggregate_Commute * yrs_risk_interact
#  * agg_commute_yrs_lifediff_int = Aggregate_Commute * yrs_lifediff_interact
#  * agg_commute_yrs_fire_int = Aggregate_Commute * yrs_fire_interact
#  * agg_commute_yrs_building_int = Aggregate_Commute * yrs_building_interact
# ##### CRIME
#  * Felony_Offense_Ratio
#  * Violent_Crime_Ratio
#  * Burglary_Larceny_Ratio
#  * Non_Intrusive_Larceny_Ratio
#  * Other_Crime_Ratio
# ##### LAND CLASSIFICATION 
#  * flood_risk_high_1
#  * flood_risk_high_2
#  * flood_risk_medium_1
#  * flood_risk_medium_2
#  * flood_risk_low_1
#  * flood_risk_low_2 
# ##### LAND USE  
#  * Commercial_LU_Ratio
#  * Industrial_LU_Ratio
#  * Residential_LU_Ratio
#  * PaddyFields_LU_Ratio
#  * OrdinaryFields_LU_Ratio
#  * Forests_LU_Ratio
#  * Pondsandswamps_LU_Ratio
#  * Misc_LU_Ratio 
# ##### DISTRICT USE 
#  * Residentail_district_ratio
#  * OtherUse_district_ratio
#  * Parks_district_ratio
#  * Unused_district_ratio
#  * Roads_district_ratio
#  * Farm_district_ratio
#  * Water_district_ratio
#  * WoodsForest_district_ratio
#  * Fields_district_ratio
# ##### POLLUTION COMPLAINTS 
#  * Air_Pollution_Complaints_per_Person 
#  * Water_Pollution_Complaints_per_Person
#  * Soil_Pollution_Complaints_per_Person
#  * Noise_Pollution_Complaints_per_Person
#  * Vibration_Complaints_per_Person
#  * Offensive_odors_Complaints_per_Person
#  * Other_Pollution_Complaints_per_Person
#  
#  
# * Aggregate Features 
#  * min_target_per_m
#  * median_target_per_m 
#  * mean_target_per_m 
#  * max_target_per_m
#  * std_target_per_m 
#  * min_target_per_m_landc
#  * median_target_per_m_landc
#  * mean_target_per_m_landc
#  * max_target_per_m_landc
#  * std_target_per_m_landc
#  
#  
#  * Category Bins 
#   * Station_1 - Binned Minority Stations to one group 
#   * Station_2 - Binned Minority Stations to one group 
#   * Station_3 - Binned Minority Stations to one group 
#   * Line_1 - Binned Minority Lines to one group 
#   * Line_2 - Binned Minority Lines to one group 
#   * Line_3 - Binned Minority Lines to one group 
#   * Location - Binned Minority locations to groups by Ku 
#   * Region - Created Categories based on side of Tokyo
#   
# 
# 
# 
# * Row Stats 
#  * row_stats_mean
#  * row_stats_std
#  * row_stats_median
#  * row_stats_min
#  * row_stats_max
#  * row_stats_zeros
#  * row_stats_negatives
#  * row_stats_kurtosis
# 
# mostcommonlayout
# firstfloor
# 
# * Target Mean Encoding - Performed on all Categorical Features 
#  * Target Encoding- location
#  * Target Encoding- Ku
#  * Target Encoding- Line_1
#  * Target Encoding- Station_1
#  * Target Encoding- Line_2
#  * Target Encoding- Station_2
#  * Target Encoding- Line_3
#  * Target Encoding- Station_3
#  * Target Encoding- Land Classification
#  * Target Encoding- floor_plan_most_common_FE
#  * Target Encoding- Region
#  
# * Geographic Feature Engineering  - Cartesian Coordinates
#   * x_cart
#   * y_cart 
#   * z_cart
# 
#   * x_cart_Eki1 
#   * y_cart_Eki1
#   * z_cart_Eki1
# 
#   * x_cart_Eki2
#   * y_cart_Eki2
#   * z_cart_Eki2
# 
#   * x_cart_Eki3
#   * y_cart_Eki3
#   * z_cart_Eki3
#  
#  
#  * Crime_Rank 
#  * Felony_Rank
#  * Violent_Crime_Rank
#  * Burglary_Larceny_Rank
#  * Non_Intrusive_Larceny_Rank
#  * Other_Crime_Rank

# In[161]:


## Create Category for Most common layout in region 
mostcommonlayout = finaldf.groupby(['location'])['floor_plan'].apply(lambda x: x.mode()).reset_index()
mostcommonlayout.rename(columns={'floor_plan':'floor_plan_most_common_FE'}, inplace=True)
## For now take level 0 - Maybe convert anything with level 1 to a multiple class for the algorithm 
mostcommonlayout = mostcommonlayout[mostcommonlayout['level_1'] == 0]
mostcommonlayout.drop(columns=['level_1'], inplace=True)

finaldf = finaldf.merge(mostcommonlayout.set_index('location'), left_on ='location', right_index=True)


# ##### Expansion Encoding

# In[162]:


## 5/14 Create Room Features for Living, Dining, Kitchen, Service_Room, and 1RM -  
finaldf['Living_FE'] = np.where(finaldf['floor_plan'].str.contains('L') == True,1,0)
finaldf['Dining_FE'] = np.where(finaldf['floor_plan'].str.contains('D') == True,1,0)
finaldf['Kitchen_FE'] = np.where(finaldf['floor_plan'].str.contains('K') == True,1,0)
finaldf['Service_Room_FE'] = np.where(finaldf['floor_plan'].str.contains('S') == True,1,0)
finaldf['Room_Only_FE'] = np.where(finaldf['floor_plan'].str.contains('RM') == True,1,0)


# In[163]:


## Remove Strings from floor_plan
finaldf['floor_plan'] = finaldf['floor_plan'].str.replace('L','')
finaldf['floor_plan'] = finaldf['floor_plan'].str.replace('D','')
finaldf['floor_plan'] = finaldf['floor_plan'].str.replace('K','')
finaldf['floor_plan'] = finaldf['floor_plan'].str.replace('S','')
finaldf['floor_plan'] = finaldf['floor_plan'].str.replace('RM','')


# In[164]:


## rename floor_plan to total rooms
finaldf.rename(columns={'floor_plan':'total_rooms_FE'}, inplace=True)


# In[165]:


## Convert total_rooms to int
finaldf['total_rooms_FE'] = finaldf['total_rooms_FE'].astype('int')


# #### Interaction Features 

# In[166]:


## Area per Room ## INTERACTION 
finaldf['area_per_room_FE'] = finaldf['area'] / finaldf['total_rooms_FE']


# In[167]:


## 5/15 ## INTERACTION 
## Ratio of Floor Living on compared to total building size
finaldf['height_ratio_FE'] = finaldf['floor'].astype(int) / finaldf['Height']

## Interaction Features - Max Corrs - Highest Correlated Features Multiplied 
finaldf['area_height_FE'] = finaldf['area'] * finaldf['Height']

## If Living on First Floor 
finaldf['first_floor_FE'] = np.where(finaldf['floor'] == 1,1,0)

## Interaction feature-  floor vs area 
finaldf['floor_area_FE'] = finaldf['floor'].astype(int) * finaldf['area']


# In[168]:


## 5/17  - Unfortunately Building material Data doesnt exist so building risk will need to be modeling in other ways

## Ideas from this paper http://hermes-ir.lib.hit-u.ac.jp/rs/bitstream/10086/13412/1/0100701601.pdf 
## http://japanpropertycentral.com/real-estate-faq/earthquake-building-codes-in-japan/ 

## Interaction between Age of Building / Risk Rating 
finaldf['yrs_risk_FE'] =  finaldf['yrs'] * finaldf['Total_Risk']

## Interaction between Age of Building / LifeDifficulty Rating
finaldf['yrs_lifediff_FE'] = finaldf['yrs'] * finaldf['Life_Difficulty_Risk']
## Interaction between Age of Building / Fire Rating 
finaldf['yrs_fire_FE'] = finaldf['yrs'] * finaldf['Fire_Risk']

## Interaction between Age of Building / Building Rating 
finaldf['yrs_building_FE'] = finaldf['yrs'] * finaldf['Building_Risk']
                                                            


# ##### Binned Features

# In[169]:


### Done #### 
## Group Minority Administrative Districts into 1 classification 
## Minority class bins 

locationgroup = finaldf.groupby(['Ku','location'])['title'].count()
locationgroup = locationgroup.reset_index() 
locationgroupminority = locationgroup[locationgroup.title < 5]

for i in locationgroupminority.Ku.unique():
  group = locationgroupminority['location'][locationgroupminority.Ku == i]
  finaldf['location'] = np.where(finaldf['location'].isin(group), str(i+'_Minority'),finaldf['location'])


# In[170]:


## Group Minority Lines Together into 1 classification 
linegroup = finaldf.groupby(['Line_1'])['title'].count()
linegroup = linegroup.reset_index() 
linegroupminority = linegroup[linegroup.title < 30]
finaldf['Line_1'] = np.where(finaldf['Line_1'].isin(linegroupminority['Line_1']), str('Line_Minority'),finaldf['Line_1'])


# In[171]:


## Group Minority Stations Together into 1 classification 
stationgroup = finaldf.groupby(['Station_1'])['title'].count()
stationgroup = stationgroup.reset_index() 
stationgroupminority = stationgroup[stationgroup.title < 30]
finaldf['Station_1'] = np.where(finaldf['Station_1'].isin(stationgroupminority['Station_1']), str('Station_Minority'),finaldf['Station_1'])


# In[172]:


## Group Minority Line 2 Together into 1 classification 
linegroup2 = finaldf.groupby(['Line_2'])['title'].count()
linegroup2 = linegroup2.reset_index() 
linegroupminority2 = linegroup2[linegroup2.title < 30]
finaldf['Line_2'] = np.where(finaldf['Line_2'].isin(linegroupminority2['Line_2']), str('Line_Minority'),finaldf['Line_2'])


# In[173]:


## Group Minority Station 2 Together into 1 classification 
stationgroup2 = finaldf.groupby(['Station_2'])['title'].count()
stationgroup2 = stationgroup2.reset_index() 
stationgroupminority2 = stationgroup2[stationgroup2.title < 30]
finaldf['Station_2'] = np.where(finaldf['Station_2'].isin(stationgroupminority2['Station_2']), str('Station_Minority'),finaldf['Station_2'])


# In[174]:


## Group Minority Line 3 Together into 1 classification 
linegroup3 = finaldf.groupby(['Line_3'])['title'].count()
linegroup3 = linegroup3.reset_index() 
linegroupminority3 = linegroup3[linegroup3.title < 30]
finaldf['Line_3'] = np.where(finaldf['Line_3'].isin(linegroupminority3['Line_3']), str('Line_Minority'),finaldf['Line_3'])


# In[175]:


## Group Minority Station 3 Together into 1 classification 
stationgroup3 = finaldf.groupby(['Station_3'])['title'].count()
stationgroup3 = stationgroup3.reset_index() 
stationgroupminority3 = stationgroup3[stationgroup3.title < 30].sort_values(by='title')
finaldf['Station_3'] = np.where(finaldf['Station_3'].isin(stationgroupminority3['Station_3']), str('Station_Minority'),finaldf['Station_3'])


# #### Transit Interactions

# In[176]:


## Total Time for All Hubs - Added together 
finaldf['Hub_Aggregate_FE'] = finaldf['ShinjukuMade'] + finaldf['TokyoMade'] + finaldf['ShibuyaMade'] + finaldf['IkebukuroMade'] + finaldf['UenoMade'] + finaldf['ShinagawaMade']


# In[177]:


## Calculate Commute Time - Walk + Transit ## INTERACTION FEATURES 
finaldf['Shinjuku_Commute_FE'] = finaldf['Time_to_Station1'] + finaldf['ShinjukuMade']
finaldf['Tokyo_Commute_FE'] = finaldf['Time_to_Station1'] + finaldf['TokyoMade']
finaldf['Shibuya_Commute_FE'] = finaldf['Time_to_Station1'] + finaldf['ShibuyaMade']
finaldf['Ikebukuro_Commute_FE'] = finaldf['Time_to_Station1'] + finaldf['IkebukuroMade']
finaldf['Ueno_Commute_FE'] = finaldf['Time_to_Station1'] + finaldf['UenoMade']
finaldf['Shinagawa_Commute_FE'] = finaldf['Time_to_Station1'] + finaldf['ShinagawaMade']


# In[178]:


## Sum of all Commute Times INTERACTION FEATURE 
finaldf['Aggregate_Commute_FE'] = finaldf['Shinjuku_Commute_FE'] + finaldf['Tokyo_Commute_FE'] + finaldf['Shibuya_Commute_FE'] + finaldf['Ikebukuro_Commute_FE'] + finaldf['Ueno_Commute_FE'] + finaldf['Shinagawa_Commute_FE']


# #### More Interactions

# In[179]:


## Interacts between aggregate AGG_Commute and building/disaster related attributes -- INTERACTION FEATURES 
finaldf['agg_commute_floor_area_FE'] = finaldf['Aggregate_Commute_FE'] * finaldf['floor_area_FE']
finaldf['agg_commute_area_height_FE'] = finaldf['Aggregate_Commute_FE'] * finaldf['area_height_FE']
finaldf['agg_commute_area_per_room_FE']= finaldf['Aggregate_Commute_FE'] * finaldf['area_per_room_FE']
finaldf['agg_commute_yrs_risk_FE'] = finaldf['Aggregate_Commute_FE'] * finaldf['yrs_risk_FE']
finaldf['agg_commute_yrs_lifediff_FE'] = finaldf['Aggregate_Commute_FE'] * finaldf['yrs_lifediff_FE']
finaldf['agg_commute_yrs_fire_FE'] = finaldf['Aggregate_Commute_FE'] * finaldf['yrs_fire_FE']
finaldf['agg_commute_yrs_building_FE'] = finaldf['Aggregate_Commute_FE'] * finaldf['yrs_building_FE']


# #### Consolidation Encoding

# In[180]:


### Done #### 
## Create Categorical Features for District of Tokyo - EXPANSION ENCODING ? 
finaldf['Region'] = np.where(finaldf['Ku'].isin(['Itabashi','Kita']),'North_outer',None)
finaldf['Region'] = np.where(finaldf['Ku'].isin(['Nerima','Suginami','Nakano']),'West',finaldf['Region'])
finaldf['Region'] = np.where(finaldf['Ku'].isin(['Toshima','Bunkyo']),'North_inner',finaldf['Region'])
finaldf['Region'] = np.where(finaldf['Ku'].isin(['Arakawa','Adachi','Katsushika','Edogawa']),'East_outer',finaldf['Region'])
finaldf['Region'] = np.where(finaldf['Ku'].isin(['Koto','Taito','Sumida']),'East_inner',finaldf['Region'])
finaldf['Region'] = np.where(finaldf['Ku'].isin(['Shibuya','Shinjuku','Minato','Chiyoda','Chuo']),'Central',finaldf['Region'])
finaldf['Region'] = np.where(finaldf['Ku'].isin(['Setagaya','Meguro','Shinagawa','Ota']),'South',finaldf['Region'])


# In[181]:


######## CRIME ######### 
##  Ranking / INTERACTION FEATURES 

## Python create overall Crime ranks 
finaldf['Crime_Rank_FE'] = finaldf['Total_Crimes'].rank(method='dense', ascending=False)

## Felonies 
finaldf['Felony_Offense_Ratio_FE'] = finaldf['Felonious_Offense_Total'] / finaldf['Total_Crimes'] 
finaldf['Felony_Rank_FE'] = finaldf['Felonious_Offense_Total'].rank(method='dense', ascending=False)
## Violent Crime Rate - Violent Crime Rank 
finaldf['Violent_Crime_Ratio_FE'] = finaldf['Violent_Crime_Total'] / finaldf['Total_Crimes'] 
finaldf['Violent_Crime_Rank_FE'] = finaldf['Violent_Crime_Total'].rank(method='dense', ascending=False)

## Burglary Larceny Rate - Burglary Larceny Rank 
finaldf['Burglary_Larceny_Ratio_FE'] = finaldf['Burglary_Larceny_Total'] / finaldf['Total_Crimes'] 
finaldf['Burglary_Larceny_Rank_FE'] = finaldf['Burglary_Larceny_Total'].rank(method='dense', ascending=False)

## Non_Intrusive Larceny Rate - Non_Intrusive_Larceny_Rank
finaldf['Non_Intrusive_Larceny_Ratio_FE'] = finaldf['Non_Intrusive_Larceny'] / finaldf['Total_Crimes'] 
finaldf['Non_Intrusive_Larceny_Rank_FE'] = finaldf['Non_Intrusive_Larceny'].rank(method='dense', ascending=False)

## Others Rate - Others Rank 
finaldf['Other_Crime_Ratio_FE'] = finaldf['Others_Total'] / finaldf['Total_Crimes'] 
finaldf['Other_Crime_Rank_FE'] = finaldf['Others_Total'].rank(method='dense', ascending=False)


# In[182]:


######### LAND USE ########
## INTERACTION FEATURES 

## Land Use 2 - Calculate Percentage of Land by Ku 
finaldf['Residentail_land_ratio_FE'] = finaldf['Residentail'] / finaldf['Total_District_Land']
finaldf['OtherUse_land_ratio_FE'] = finaldf['OtherUse'] / finaldf['Total_District_Land']
finaldf['Parks_land_ratio_FE'] = finaldf['Parks'] / finaldf['Total_District_Land']
finaldf['Unused_land_ratio_FE'] = finaldf['Unused'] / finaldf['Total_District_Land']
finaldf['Roads_land_ratio_FE'] = finaldf['Roads'] / finaldf['Total_District_Land']
finaldf['Farm_land_ratio_FE'] = finaldf['Farm'] / finaldf['Total_District_Land']
finaldf['Water_land_ratio_FE'] = finaldf['Water'] / finaldf['Total_District_Land']
finaldf['WoodsForest_land_ratio_FE'] = finaldf['WoodsForest'] / finaldf['Total_District_Land']
finaldf['Fields_land_ratio_FE'] = finaldf['Fields'] / finaldf['Total_District_Land']


# In[183]:


######## POLLUTION COMPLAINTS ###########
## INTERACTION FEATURES 

## Pollution Complaints - Calculate Pollution complaints by population 
finaldf['Air_Pollution_Complaints_per_Person_FE'] = finaldf['Air pollution'] / finaldf['Population']
finaldf['Water_Pollution_Complaints_per_Person_FE'] = finaldf['Water pollution'] / finaldf['Population']
finaldf['Soil_Pollution_Complaints_per_Person_FE'] = finaldf['Soil pollution'] / finaldf['Population']
finaldf['Noise_Pollution_Complaints_per_Person_FE'] = finaldf['Noise'] / finaldf['Population']
finaldf['Vibration_Complaints_per_Person_FE'] = finaldf['Vibration'] / finaldf['Population']
finaldf['Offensive_odors_Complaints_per_Person_FE'] = finaldf['Offensive_odors'] / finaldf['Population']
finaldf['Other_Pollution_Complaints_per_Person_FE'] = finaldf['other_pollution'] / finaldf['Population']


# In[184]:


#### Cartesian Projections #### 
## Cartesian Projection 
finaldf['x_cart_FE'] = np.cos(finaldf['LAT']) * np.cos(finaldf['LON'])
finaldf['y_cart_FE'] = np.cos(finaldf['LAT']) * np.sin(finaldf['LON'])
finaldf['z_cart_FE'] = np.sin(finaldf['LAT'])

## Cartesian Projection Station 1 
finaldf['x_cart_Eki1_FE'] = np.cos(finaldf['LAT_Eki1']) * np.cos(finaldf['LON_Eki1'])
finaldf['y_cart_Eki1_FE'] = np.cos(finaldf['LAT_Eki1']) * np.sin(finaldf['LON_Eki1'])
finaldf['z_cart_Eki1_FE'] = np.sin(finaldf['LAT_Eki1'])

## Cartesian Projection Station 2 
finaldf['x_cart_Eki2_FE'] = np.cos(finaldf['LAT_Eki2']) * np.cos(finaldf['LON_Eki2'])
finaldf['y_cart_Eki2_FE'] = np.cos(finaldf['LAT_Eki2']) * np.sin(finaldf['LON_Eki2'])
finaldf['z_cart_Eki2_FE'] = np.sin(finaldf['LAT_Eki2'])

## Cartesian Projection Station 3 
finaldf['x_cart_Eki3_FE'] = np.cos(finaldf['LAT_Eki3']) * np.cos(finaldf['LON_Eki3'])
finaldf['y_cart_Eki3_FE'] = np.cos(finaldf['LAT_Eki3']) * np.sin(finaldf['LON_Eki3'])
finaldf['z_cart_Eki3_FE'] = np.sin(finaldf['LAT_Eki3'])


# ### Target Encoding

# In[185]:


## TARGET ENCODING 
train = finaldf[finaldf['train'] == 1]
valid = finaldf[finaldf['valid'] == 1]
test = finaldf[finaldf['test'] == 1]
  
  ## n_train shape, n_valid shape, n_test shape 
n_train = train.shape[0]
n_valid = valid.shape[0]
n_test = test.shape[0]
  
  ## Split Train, Valid, Test Set 
y_train = train.target
y_valid = valid.target
y_test = test.target
X_train = train.drop(columns='target')
X_valid = valid.drop(columns='target')
X_test = test.drop(columns='target')


# In[186]:


categoricalcolumns = X_train.select_dtypes(['object','category']).drop(columns=['title','Ku','Land Classification','Region','floor_plan_most_common_FE']).columns


# In[187]:




## 10 Location
print(np.percentile(X_train.groupby('location').agg('count')['title'].sort_values(),20),'percentile for counts')

## 85 - Line 1
print(np.percentile(X_train.groupby('Line_1').agg('count')['title'].sort_values(),20),'percentile for counts')
## 55 - Station 1
print(np.percentile(X_train.groupby('Station_1').agg('count')['title'].sort_values(),20),'percentile for counts')
## 125 - Line_2 
print(np.percentile(X_train.groupby('Line_2').agg('count')['title'].sort_values(),20),'percentile for counts')
## 50 - Station_2 
print(np.percentile(X_train.groupby('Station_2').agg('count')['title'].sort_values(),20),'percentile for counts')
## 50 - Line_3
print(np.percentile(X_train.groupby('Line_3').agg('count')['title'].sort_values(),20),'percentile for counts')
## 45 - Station_3 
print(np.percentile(X_train.groupby('Station_3').agg('count')['title'].sort_values(),20),'percentile for counts')


# In[188]:


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(train=None, 
                  valid=None,
                  test=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    #assert len(train) == len(target)
    #assert train.name == test.name
    temp = pd.concat([train, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=train.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
   ### TRAIN 
    ft_trn_series = pd.merge(
        train.to_frame(train.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=train.name,
        how='left')['average'].rename(train.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = train.index 
    
    ### VALID 
    ft_valid_series = pd.merge(
        valid.to_frame(valid.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=valid.name,
        how='left')['average'].rename(train.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_valid_series.index = valid.index 
    
    ft_tst_series = pd.merge(
        test.to_frame(test.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=test.name,
        how='left')['average'].rename(train.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = test.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_valid_series, noise_level), add_noise(ft_tst_series, noise_level)


# In[189]:


## Smoothing = 5 - 5.45
## Noise .001 - 5.31 
## Noise .0001 - 5.32 

## 5.32 10th percentile samples
## 5.22 20th percentile samples 

## 5.25 smoothing = 100
## 5.18 smoothing = 10 
## ???  smoothing = 8 

## SMoothing = 4 - 
## Smoothing = 3 - 
## Smoothing = 2 - 
## Smoothing = 1 - 5.44

#samples = [10,85,55,125,50,50,45]
samples = [15,152,87,300,92,278,78]
#samples = [20,300, 120, 300, 129, 300, 119]
for category, sample in zip(categoricalcolumns,samples):
  #print('Min Samples for Leaf-',sample)
  X_train[category], X_valid[category],X_test[category] = target_encode(train=X_train[category], 
                  valid=X_valid[category],
                  test=X_test[category], 
                  target=y_train, 
                  min_samples_leaf=sample, 
                  smoothing=10,
                  noise_level=.001)
  print('Target Encoding-',category)


# In[190]:


target = pd.concat([y_train,y_valid,y_test])


# In[191]:


finaldf = pd.concat([X_train,X_valid, X_test])


# In[192]:


finaldf['target'] = target


# In[193]:


finaldf.rename(columns={'location':'location_FE',
                        'Line_1':'Line_1_FE',
                        'Station_1':'Station_1_FE',
                        'Line_2':'Line_2_FE',
                        'Station_2':'Station_2_FE',
                        'Line_3':'Line_3_FE',
                        'Station_3':'Station_3_FE'}, inplace=True)


# In[195]:


## Return Correlations 
print(finaldf.corr()['target'].sort_values())


# In[196]:


X_train, X_valid, X_test, allcolumns, y_train, y_valid, y_test = pre_process(finaldf,['title','rent'])


# ## Create Data - Feature Engineering Model

# In[199]:


## Save Training, Valid, and Test Files 
## Save Training, Valid, and Test Files 
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/train')

sparse.save_npz('X_train_FE.npz', X_train)
y_train.to_hdf('y_train_FE.hdf', 'train')

os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/valid')
sparse.save_npz('X_valid_FE.npz', X_valid)
y_valid.to_hdf('y_valid_FE.hdf', 'valid')

os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/test')
sparse.save_npz('X_test_FE.npz', X_test)
y_test.to_hdf('y_test_FE.hdf','test')


## Columns for LGBM 
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/')
pd.DataFrame(allcolumns).to_csv('columns_FE.csv',index=False)

