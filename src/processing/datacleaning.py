## Import Libraries

## Shuffle
import os
from sklearn.utils import shuffle

## Basic Data Processing
import pandas as pd
import numpy as np

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

plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.5)

## Change to Data Directory
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/raw')

## Read Suumo Data
print('Read Suumo Dataset')
suumo = pd.read_csv('tokyo.csv')

## remove unwanted characters
suumo.floor = suumo.floor.str.replace('階', '').str.replace('\r\n\t\t\t\t\t\t\t\t\t\t\t', '')

# shuffle data
suumo = shuffle(suumo, random_state=13)

## Get Total Samples
totalsamples = suumo.shape[0]

# ## Create Target Value
print('Creating Target Value')

## Rent - Pre-Processing
suumo.rent = suumo.rent.str.strip('万円').astype(float) * 10000

## Admin Fee
## - strip manen and add 0's
suumo.admin.replace('-', 0, inplace=True)
suumo.admin = suumo.admin.str.strip('円').astype(float)
## Fill NA's with 0
suumo.admin.fillna(0, inplace=True)
## create admin flag
suumo['admin_flag'] = np.where(suumo['admin'] > 0, 1, 0)

## Create Target Variable (Rent + Admin Fee)
suumo['target'] = suumo['rent'] + suumo['admin']

## Drop Admin Fee - Not needed in this exercise
suumo.drop(columns=['admin'], inplace=True)

## Outlier- Removal Noise
print('Begin Outlier Removal Process - Suumo Data')
totalsamples = suumo.shape[0]

print('Total Samples before Outlier Removal', totalsamples)

## Drop Target above 800,000 Yen - Outliers Removal - Couldn't get distributions to match
## Removed Rent below 23000 as apartments below this price are extreme outliers
suumo = suumo[suumo['target'] < 800000]
suumo = suumo[suumo['target'] > 23000]

print('Outliers Removed Target > 800000 or < 23000', totalsamples - suumo.shape[0])
totalsamples = suumo.shape[0]

# #### Clean up Area/ Yrs
print('Clean up Area/Yrs field pre')
## Area - textcleanup- Remove meters squared
suumo.area = suumo.area.str.replace('m2', '')

## Years text cleanup
## Create new column for new building
suumo.yrs = suumo.yrs.str.strip('築').str.strip('年')
suumo['new'] = np.where(suumo.yrs == '新', 1, 0)
suumo.yrs.replace('新', '0', inplace=True)

## Type Conversion
suumo.area = suumo.area.astype('float')
suumo.yrs = suumo.yrs.astype('int')

# ## OUTLIER HANDLING - Rent Per M by Year
## Calculate rent_per_m
suumo['rent_per_m'] = suumo['target'] / suumo['area']

## Drop 99 years - Looks like noise
suumo = suumo[suumo['yrs'] != 99]

## Yr Bins -- Create Bins for years above 50 as the samples dwindle
suumo['yrbins'] = np.where((suumo['yrs'] >= 50) & (suumo['yrs'] < 60), '50s', suumo['yrs'])
suumo['yrbins'] = np.where((suumo['yrs'] >= 60) & (suumo['yrs'] < 70), '60s', suumo['yrs'])
suumo['yrbins'] = np.where((suumo['yrs'] >= 70) & (suumo['yrs'] < 80), '70s', suumo['yrbins'])
suumo['yrbins'] = np.where((suumo['yrs'] >= 80) & (suumo['yrs'] < 90), '80s', suumo['yrbins'])
suumo['yrbins'] = np.where((suumo['yrs'] >= 90) & (suumo['yrs'] < 100), '90s', suumo['yrbins'])

## Detect Outliers
outlierdetect = suumo.groupby(['Ku', 'yrbins'])['rent_per_m'].agg(['count', 'mean', 'min', 'max'])

## Merge outlier detection to Ku
suumo = suumo.merge(outlierdetect, how='left', left_on=['Ku', 'yrbins'], right_index=True)

## Create flag for upper outlier -
suumo['upper_outlier'] = np.where((suumo['mean'] * 5) < suumo['rent_per_m'], 1, 0)

## Create flag for lower outlier
suumo['lower_outlier'] = np.where(suumo['rent_per_m'] < (suumo['mean'] / 5), 1, 0)

## lower_outlier - REMOVE
suumo = suumo[suumo['lower_outlier'] != 1]

## upper_outlier - REMOVE
suumo = suumo[suumo['upper_outlier'] != 1]

## Column Cleanup- Remove garbage columns
suumo.drop(columns=['rent_per_m', 'count', 'mean', 'min', 'max', 'upper_outlier', 'lower_outlier', 'yrbins'],
           inplace=True)

print('Outliers Removed with high price compared to rent per m and age', totalsamples - suumo.shape[0])
totalsamples = suumo.shape[0]

# ## Outlier Handling - Room Layout
## Deletion of Values with Ridiculous Room Layouts like 22LDK or 44k
suumo = suumo[~suumo.floor_plan.isin(['22LDK', '12K'])]

print('Outliers Removed due to ridiculous room layout-', totalsamples - suumo.shape[0])
totalsamples = suumo.shape[0]

print('Total Samples after Outlier Removal', totalsamples)

## Train/Test Split

print('Split Train/Test/Valid data into 15 stratified bins based on target value')
## Split Train/Test - 15 Bins
suumo['target_bins'] = pd.qcut(suumo['target'], 15)
train, test = train_test_split(suumo, stratify=suumo['target_bins'], test_size=.2, random_state=13)

## Split Train/Valid
train, valid = train_test_split(train, stratify=train['target_bins'], test_size=.2, random_state=13)

print('Price bins-', np.array(suumo['target_bins'].unique().sort_values()))

## Indexes
train_in = train.index.values
valid_in = valid.index.values
test_in = test.index.values

## Combine Train and Test
print('Re-combine data and create flags for train, valid, and test set')
suumo = pd.concat([train, valid, test], axis=0)

## Create Column indicators
suumo['train'] = np.where(suumo.index.isin(train_in), 1, 0)
suumo['valid'] = np.where(suumo.index.isin(valid_in), 1, 0)
suumo['test'] = np.where(suumo.index.isin(test_in), 1, 0)

suumo.drop(columns=['target_bins'], inplace=True)

print('Plot distributions for Train/Valid/Test Set')

## Plot Distribution for Train vs Validation Set
fig, ax = plt.subplots(ncols=3, figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.title('Train vs Validation Set', weight='bold')
plt.hist(suumo[suumo['train'] == 1]['target'], bins=25, rwidth=0.8, density=True,
         range=[suumo['target'].min(), suumo['target'].max()], alpha=.6, color='teal', label='Train', log=True)
plt.hist(suumo[suumo['valid'] == 1]['target'], bins=25, rwidth=0.8, density=True,
         range=[suumo['target'].min(), suumo['target'].max()], alpha=.6, color='crimson', label='Valid', log=True)
plt.legend()

## Plot Distribution for Train vs Test Set
plt.subplot(1, 3, 2)
plt.title('Train vs Validation Set', weight='bold')
plt.hist(suumo[suumo['train'] == 1]['target'], bins=25, rwidth=0.8, density=True,
         range=[suumo['target'].min(), suumo['target'].max()], alpha=.6, color='teal', label='Train', log=True)
plt.hist(suumo[suumo['test'] == 1]['target'], bins=25, rwidth=0.8, density=True,
         range=[suumo['target'].min(), suumo['target'].max()], stacked=True, alpha=.6, color='purple', label='Test',
         log=True)
plt.legend()

## Plot Distribution for Train + Valid vs Test Set
plt.subplot(1, 3, 3)
plt.title('Train & Valid vs Test Set', weight='bold')
plt.hist(suumo[(suumo['train'] == 1) | (suumo['valid'] == 1)]['target'], bins=25, rwidth=0.8, density=True,
         range=[suumo['target'].min(), suumo['target'].max()], alpha=.6, color='orange', label='Train+Valid', log=True)
plt.hist(suumo[suumo['test'] == 1]['target'], bins=25, rwidth=0.8, density=True,
         range=[suumo['target'].min(), suumo['target'].max()], stacked=True, alpha=.6, color='purple', label='Test',
         log=True)

plt.legend()
plt.tight_layout()
fig.savefig('distributions.png')

## Diagnostics Distribution of Target for each dataset
print('print distributions for each dataset')
traindist = pd.DataFrame(train['target'].describe()).T.rename(index={'target': 'train'})
validdist = pd.DataFrame(valid['target'].describe()).T.rename(index={'target': 'valid'})
testdist = pd.DataFrame(test['target'].describe()).T.rename(index={'target': 'test'})

## Print Diagnostics
datasetstats = pd.concat([traindist, validdist, testdist])
print(datasetstats)

## Pre-Processing Steps
print('Begin feature pre-proc')

## Log Transform Area
suumo['area'] = np.log(suumo['area'])

## Gratuity - strip manen and add 0's
suumo.gratuity = suumo.gratuity.str.strip('万円')
suumo.gratuity.replace('-', 0, inplace=True)
suumo.gratuity = suumo.gratuity.astype(float) * 10000

## Create Gratuity flag
suumo.gratuity = np.where(suumo.gratuity > 0, 1, 0)

## Deposit - strip manen and add 0's
suumo.deposit = suumo.deposit.str.strip('万円')
suumo.deposit.replace('-', 0, inplace=True)
suumo.deposit = suumo.deposit.astype(float) * 10000

## Create Deposit Flag
suumo.deposit = np.where(suumo.deposit > 0, 1, 0)

## Location -- Drop Tokyo from address
suumo['location'] = suumo['location'].str.replace('東京都', '')
## Imputations to match inconsistent Data - Crime Data - Need to impute these to align joins
suumo['location'] = suumo['location'].str.replace("港区元赤坂２", "港区元麻布２")
suumo['location'] = suumo['location'].str.replace("足立区西伊興町", "足立区西伊興２")
suumo['location'] = suumo['location'].str.replace("江戸川区鹿骨町", "江戸川区鹿骨２")
## Imputes to match inconsistent Data - Disaster Data - Need to impute these to align joins
suumo['location'] = suumo['location'].str.replace("新宿区三栄町", "新宿区四谷三栄町")

## Floor Plan
suumo.floor_plan.replace('ワンルーム', '1RM', inplace=True)

## Height - Split To Two Columns - Above ground and under ground
height = suumo.heights.str.split('地上', expand=True)
height[1] = np.where(height[1].isnull() == True, height[0], height[1])

height[0] = height[0].str.strip('地下')
height[0] = np.where(height[0].str.contains('階建'), 0, height[0])
height[0] = np.where(height[0].str.contains('平屋'), 0, height[0])

# 平屋- Means 1 floor
height[1] = height[1].str.strip('階建')
height[1] = height[1].replace('平屋', '1')

height.columns = ['Basement_Depth', 'Height']

## Add new height columns back to suumo data
suumo = pd.concat([suumo, height], axis=1).drop(columns='heights')

## Create features based on Floors

## Entire House Feature
suumo['entirehouse'] = np.where(suumo.floor == '-', 1, 0)

## Mezanine feature
suumo['mezanine'] = np.where(suumo.floor.isin(['M2', 'M1', 'M3']), 1, 0)

## Multifloor Feature
multifloors = ['2-1', '1-3', '9-10', '4-5', '3-1', '3-2', '2-3', '8-9', '3-4', 'B1-3', '1-B1', 'B1-1', '1-4', '5-6',
               '3-5', '7-8', '37-38', '12-13', '12-11', '10-11', '1-1', '2-2', '6-7'
                                                                               '2-5', 'B1-2', '46-47', '13-14', '14-15',
               '2-4', 'B2-1', '2-B1', '11-12']

suumo['multifloor'] = np.where(suumo.floor.isin(multifloors), 1, 0)

## Clean Floors - In the case of Multi floor- Highest Floor is Selected, Negative floor indicates highest floor is basement
suumo['floor'] = suumo.floor.replace(
    {'2-1': '2', '1-3': '3', '9-10': '10', '4-5': '5', '3-1': '3', '3-2': '3', '2-3': '3', '8-9': '9', '3-4': '4',
     'B1-3': '-1', '1-B1': '1', 'B1-1': '1', '1-4': '4', '5-6': '6', '3-5': '5', '7-8': '8', '37-38': '38',
     '12-13': '13', '12-11': '12', '10-11': '11', '1-1': '1', '2-2': '2', '6-7': '7', '2-5': '5', 'B1-2': '-1',
     '46-47': '47', '13-14': '14', '14-15': '15', '2-4': '4', 'B2-1': '-1', '2-B1': '2', '11-12': '12',
     '1-2': '2', 'B1': '-1', 'B2': '-2', 'M2': '2', 'M1': '1', 'M3': '3', 'B14': '14', 'B10': '10', 'B3': '3',
     'B8': '8', 'B13': '13', 'B4': '4', 'B': '-1', 'B5': '5'})

## - Value usually indicates entire house, height of building is used for floor
suumo.floor = np.where(suumo.floor == '-', suumo.Height, suumo.floor)

## Replace floor typos -B higher floors were encoded as positive because possible typo - Using building height and basement height as proxy
suumo['floor'] = np.where(suumo['floor'].astype(int) < suumo['Basement_Depth'].astype(int) * -1,
                          suumo['floor'].astype(int) * -1, suumo['floor'])

## Lives in basement feature
basementdwellers = ['-2', '-1']
suumo['basementdweller'] = np.where(suumo.floor.isin(basementdwellers), 1, 0)

## Type Conversion
suumo.area = suumo.area.astype('float').round(2)
suumo.Basement_Depth = suumo.Basement_Depth.astype('int')
suumo.Height = suumo.Height.astype('int')
suumo.yrs = suumo.yrs.astype('int')
suumo.rent = suumo.rent.astype('int32')
suumo.target = suumo.target.astype('int32')

## Pre-process Station 1, 2, 3

## Station 1
split = suumo['station1'].str.split(' ', expand=True)
## Drop Column 2 its garbage
split.drop(columns=2, inplace=True)
## Car
split['Car_1'] = np.where(split[1].str.contains('車'), split[1], None)
## Walk
split['Walk_1'] = np.where(split[3].isnull() == True, split[1], split[3])
split['Walk_1'] = np.where(split['Walk_1'].str.contains('車') == True, None, split['Walk_1'])
## Can drop 3 now that we have extracted all the information
split.drop(columns=3, inplace=True)

##Bus
split['Bus_1'] = np.where(split[1].str.contains('バス'), split[1], None)
split.drop(columns=1, inplace=True)

## Remove Kanji/Kana from Car, Walk, Bus
split['Walk_1'] = split['Walk_1'].str.replace('歩', '').str.replace('分', '')
split['Car_1'] = split['Car_1'].str.replace('車', '').str.split('分', expand=True)[0]
split['Bus_1'] = split['Bus_1'].str.replace('バス', '').str.replace('分', '')

## Split Station and Line
split2 = split[0].str.split('/', expand=True)
split2.columns = ['Line_1', 'Station_1']

station1info = pd.concat([split, split2], axis=1).drop(columns=0)

## Impute 1 incorrect value
station1info['Bus_1'][station1info.index == 30223] = '10'
station1info['Station_1'][station1info.index == 30223] = '葛西駅'
station1info['Line_1'][station1info.index == 30223] = station1info['Line_1'][station1info.index == 30223].str.replace('葛西駅', '')

## Station 2
## Ratio between two stations

split = suumo['station2'].str.split(' ', expand=True)
## Drop Column 2 its garbage
split.drop(columns=2, inplace=True)
## Car
split['Car_2'] = np.where(split[1].str.contains('車'), split[1], None)
## Walk
split['Walk_2'] = np.where(split[3].isnull() == True, split[1], split[3])
split['Walk_2'] = np.where(split['Walk_2'].str.contains('車') == True, None, split['Walk_2'])
## Can drop 3 now that we have extracted all the information
split.drop(columns=3, inplace=True)

##Bus
split['Bus_2'] = np.where(split[1].str.contains('バス'), split[1], None)
split.drop(columns=1, inplace=True)

## Little Cleaning
split['Walk_2'][split.index == 3082] = '歩3分'
split.drop(columns=4, inplace=True)

## Remove Kanji/Kana from Car, Walk, Bus
split['Walk_2'] = split['Walk_2'].str.replace('歩', '').str.replace('分', '')
split['Car_2'] = split['Car_2'].str.replace('車', '').str.split('分', expand=True)[0]
split['Bus_2'] = split['Bus_2'].str.replace('バス', '').str.replace('分', '')

## Split Station and Line
split2 = split[0].str.split('/', expand=True)
split2.columns = ['Line_2', 'Station_2']

station2info = pd.concat([split, split2], axis=1).drop(columns=0)

## Replace NaN Values with None- Had to create a workaround as could not easily convert NaN to None
station2info['Walk_2'] = station2info['Walk_2'].replace(np.nan, 'NaN')
station2info['Walk_2'] = np.where(station2info['Walk_2'] == 'NaN', None, station2info['Walk_2'])

station2info['Car_2'] = station2info['Car_2'].replace(np.nan, 'NaN')
station2info['Car_2'] = np.where(station2info['Car_2'] == 'NaN', None, station2info['Car_2'])

station2info['Bus_2'] = station2info['Bus_2'].replace(np.nan, 'NaN')
station2info['Bus_2'] = np.where(station2info['Bus_2'] == 'NaN', None, station2info['Bus_2'])

station2info['Line_2'] = station2info['Line_2'].replace(np.nan, 'NaN')
station2info['Line_2'] = np.where(station2info['Line_2'] == 'NaN', 'No_Line_2', station2info['Line_2'])

station2info['Station_2'] = station2info['Station_2'].replace(np.nan, 'NaN')
station2info['Station_2'] = np.where(station2info['Station_2'] == 'NaN', "No_Station_2", station2info['Station_2'])

## Station 3
split = suumo['station3'].str.split(' ', expand=True)
## Drop Column 2 its garbage
split.drop(columns=2, inplace=True)
## Car
split['Car_3'] = np.where(split[1].str.contains('車'), split[1], None)
## Walk
split['Walk_3'] = np.where(split[3].isnull() == True, split[1], split[3])
split['Walk_3'] = np.where(split['Walk_3'].str.contains('車') == True, None, split['Walk_3'])

## Can drop 3 now that we have extracted all the information
split.drop(columns=3, inplace=True)

##Bus
split['Bus_3'] = np.where(split[1].str.contains('バス'), split[1], None)
split.drop(columns=1, inplace=True)

## Remove Kanji/Kana from Car, Walk, Bus
split['Walk_3'] = split['Walk_3'].str.replace('歩', '').str.replace('分', '')
split['Car_3'] = split['Car_3'].str.replace('車', '').str.split('分', expand=True)[0]
split['Bus_3'] = split['Bus_3'].str.replace('バス', '').str.replace('分', '')

## Split Station and Line
split2 = split[0].str.split('/', expand=True)
split2.columns = ['Line_3', 'Station_3']

station3info = pd.concat([split, split2], axis=1).drop(columns=0)
station3info.head()

## Replace NaN Values with None- Had to create a workaround as could not easily convert NaN to None
station3info['Walk_3'] = station3info['Walk_3'].replace(np.nan, 'NaN')
station3info['Walk_3'] = np.where(station3info['Walk_3'] == 'NaN', None, station3info['Walk_3'])

station3info['Car_3'] = station3info['Car_3'].replace(np.nan, 'NaN')
station3info['Car_3'] = np.where(station3info['Car_3'] == 'NaN', None, station3info['Car_3'])

station3info['Bus_3'] = station3info['Bus_3'].replace(np.nan, 'NaN')
station3info['Bus_3'] = np.where(station3info['Bus_3'] == 'NaN', None, station3info['Bus_3'])

station3info['Line_3'] = station3info['Line_3'].replace(np.nan, 'NaN')
station3info['Line_3'] = np.where(station3info['Line_3'] == 'NaN', 'No_Line_3', station3info['Line_3'])

station3info['Station_3'] = station3info['Station_3'].replace(np.nan, 'NaN')
station3info['Station_3'] = np.where(station3info['Station_3'] == 'NaN', 'No_Station_3', station3info['Station_3'])

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

## Final Combine
df = pd.concat([suumo, station1info, station2info, station3info], axis=1)
## drop old features
df.drop(['station1', 'station2', 'station3'], axis=1, inplace=True)

## Update D types
df['Car_1'] = df['Car_1'].astype(float)
df['Bus_1'] = df['Bus_1'].astype(float)
df['Walk_1'] = df['Walk_1'].astype(float)
df['Car_2'] = df['Car_2'].astype(float)
df['Bus_2'] = df['Bus_2'].astype(float)
df['Walk_2'] = df['Walk_2'].astype(float)
df['Car_3'] = df['Car_3'].astype(float)
df['Bus_3'] = df['Bus_3'].astype(float)
df['Walk_3'] = df['Walk_3'].astype(float)
df['floor'] = df['floor'].astype('int32')

## Change to Data Directory and write processed file
print('Write Suumo File after pre-processing')
os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/postproc')
df.to_csv('tokyo_postproc.csv',index=False)