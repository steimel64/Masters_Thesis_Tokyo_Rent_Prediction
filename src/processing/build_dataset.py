#!/usr/bin/env python
# coding: utf-8

# Shuffle
import os
from sklearn.utils import shuffle

# Basic Data Processing
import pandas as pd
import numpy as np

# Train/Test Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

# sparse
from scipy.sparse import vstack
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix

# Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.5)


# Reduce Mem Usage
def reduce_mem_usage(props, prompt=True):
    nan_cols = props.columns[props.isnull().any()].tolist()
    if prompt:
        start_mem_usg = props.memory_usage().sum() / 1024 ** 2
        print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            if prompt:
                # Print current column type
                print("******************************")
                print("Column: ", col)
                print("dtype before: ", props[col].dtype)
            if col in nan_cols:
                if prompt:
                    print('Column: %s has NAN values' % col)
                props.loc[:, col] = props.loc[:, col].astype(np.float32)
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
                        if mx < 2 ** 8:
                            props.loc[:, col] = props.loc[:, col].astype(np.uint8)
                        elif mx < 2 ** 16:
                            props.loc[:, col] = props.loc[:, col].astype(np.uint16)
                        elif mx < 2 ** 32:
                            props.loc[:, col] = props.loc[:, col].astype(np.uint32)
                        else:
                            props.loc[:, col] = props.loc[:, col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props.loc[:, col] = props.loc[:, col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props.loc[:, col] = props.loc[:, col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props.loc[:, col] = props.loc[:, col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props.loc[:, col] = props.loc[:, col].astype(np.int64)

                # Make float datatypes 32 bit
                else:
                    props.loc[:, col] = props.loc[:, col].astype(np.float32)

            if prompt:
                # Print new column type
                print("dtype after: ", props[col].dtype)
                print("******************************")

    if prompt:
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024 ** 2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props


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
    traindist = pd.DataFrame(y_train.describe()).T.rename(index={'target': 'train'})
    validdist = pd.DataFrame(y_valid.describe()).T.rename(index={'target': 'valid'})
    testdist = pd.DataFrame(y_test.describe()).T.rename(index={'target': 'test'})

    datasetstats = pd.concat([traindist, validdist, testdist])
    print('------Data Distribution------')
    print(datasetstats)

    ## Scaling Steps- NUMERIC FEATURES
    numerics = ['uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    print(X_train.shape, X_valid.shape, X_test.shape)
    print('-----------------------')
    sc = StandardScaler()
    X_train_num = sc.fit_transform(X_train.drop(['train', 'valid', 'test'], axis=1).select_dtypes(include=numerics))
    X_valid_num = sc.transform(X_valid.drop(['train', 'valid', 'test'], axis=1).select_dtypes(include=numerics))
    X_test_num = sc.transform(X_test.drop(['train', 'valid', 'test'], axis=1).select_dtypes(include=numerics))

    numeric_column_names = X_train.drop(['train', 'valid', 'test'], axis=1).select_dtypes(include=numerics).columns
    print('Number of Numeric Columns-', numeric_column_names.shape[0])
    print('Numeric Columns-', numeric_column_names.values)
    print('-----------------------')

    ## Stack Numeric Data
    full_num_data = vstack([X_train_num, X_valid_num, X_test_num])

    ## Combine Full Data
    full_data = pd.concat([X_train, X_valid, X_test], axis=0)

    ## Collect Object Columns
    objectcolumns = full_data.select_dtypes(['object', 'category']).columns
    print('Categorical Columns=', objectcolumns.values, 'shape=', objectcolumns.shape[0])

    ## Loop LabelBinarizer
    full_object_data = csr_matrix((0, 0))
    object_column_names = np.array([])
    ## Binarizer
    lb = LabelBinarizer()
    for i in objectcolumns:
        labels = lb.fit_transform(full_data[i])
        columns = lb.classes_
        print(i, 'features-', labels.shape[1])
        full_object_data = sparse.hstack([full_object_data, labels])
        object_column_names = np.concatenate((object_column_names, columns))

    ## Stack Numeric / Object Data
    X = hstack((full_object_data, full_num_data)).tocsr()
    allcolumns = np.concatenate((object_column_names, numeric_column_names))
    print('Categorical Features Total=', X.shape[1] - numeric_column_names.shape[0])
    print('-----------------------')
    print('Full Shape =', X.shape)

    ## Create Final X_train, X_valid, and X_test
    X_train = X[:n_train]
    X_valid = X[n_train:n_train + n_valid]
    X_test = X[n_train + n_valid:]
    print('----------------------')

    print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)

    return (X_train, X_valid, X_test, allcolumns, y_train, y_valid, y_test)


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
    # assert len(train) == len(target)
    # assert train.name == test.name
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
    return add_noise(ft_trn_series, noise_level), add_noise(ft_valid_series, noise_level), add_noise(ft_tst_series,
                                                                                                     noise_level)


if __name__ == '__main__':

    ## Change to Data Directory
    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/postproc')
    df = pd.read_csv('tokyo_postproc.csv')

    # ## Create Dataset - Baseline Model
    print('------------CREATE BASELINE DATASETS---------------')
    X_train, X_valid, X_test, allcolumns, y_train, y_valid, y_test = pre_process(df, ['title', 'rent'])

    ## Save Training, Valid, and Test Files
    print('------------SAVE BASELINE DATASETS---------------')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/train')
    sparse.save_npz('X_train.npz', X_train)
    y_train.to_hdf('y_train.hdf', 'train')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/valid')
    sparse.save_npz('X_valid.npz', X_valid)
    y_valid.to_hdf('y_valid.hdf', 'valid')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/test')
    sparse.save_npz('X_test.npz', X_test)
    y_test.to_hdf('y_test.hdf', 'test')


    ## Columns for LGBM
    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/baseline_model/')
    pd.DataFrame(allcolumns).to_csv('columns.csv', index=False)

    ## GeoLocation Data (Index Reset)
    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/raw')

    ## Import Longitude Latitudes/ Location Data
    locationsref = pd.read_csv('locationspp.csv')
    ekiref = pd.read_csv('stationspp.csv')

    ## Drop Tokyo
    locationsref['location'] = locationsref.location.str.replace("東京都", "")

    ## Merge Geo Data to Locations
    df = df.merge(locationsref.set_index("location"), left_on="location", right_index=True)

    ## Merge Geo Data to Locations
    df = df.merge(ekiref.set_index("Stations"), left_on="Station_1", right_index=True, suffixes=('', '_Eki1'))
    df = df.merge(ekiref.set_index("Stations"), how='left', left_on="Station_2", right_index=True,
                  suffixes=('', '_Eki2'))
    df = df.merge(ekiref.set_index("Stations"), how='left', left_on="Station_3", right_index=True,
                  suffixes=('', '_Eki3'))

    ## Station 1 Flags - Add Car_1, Walk_1, Bus_1 - create flags
    df['Time_to_Station1'] = df.fillna(0)['Car_1'] + df.fillna(0)['Walk_1'] + df.fillna(0)['Bus_1']
    df['WalkFlag'] = np.where(df['Walk_1'] > 0, 1, 0)
    df['BusFlag'] = np.where(df['Bus_1'] > 0, 1, 0)
    df['CarFlag'] = np.where(df['Car_1'] > 0, 1, 0)

    ## Station 2 Flags - Add Car_2, Walk_2, Bus_2 - create flags
    df['Time_to_Station2'] = df.fillna(0)['Car_2'] + df.fillna(0)['Walk_2'] + df.fillna(0)['Bus_2']
    df['Time_to_Station2'] = np.where(df['Time_to_Station2'] == 0, None, df['Time_to_Station2']).astype('float32')
    df['WalkFlag_2'] = np.where(df['Walk_2'] > 0, 1, 0)
    df['BusFlag_2'] = np.where(df['Bus_2'] > 0, 1, 0)
    df['CarFlag_2'] = np.where(df['Car_2'] > 0, 1, 0)

    ## Station 3 Flags - Add Car_3, Walk_3, Bus_3 - create flags
    df['Time_to_Station3'] = df.fillna(0)['Car_3'] + df.fillna(0)['Walk_3'] + df.fillna(0)['Bus_3']
    df['Time_to_Station3'] = np.where(df['Time_to_Station3'] == 0, None, df['Time_to_Station3']).astype('float32')
    df['WalkFlag_3'] = np.where(df['Walk_3'] > 0, 1, 0)
    df['BusFlag_3'] = np.where(df['Bus_3'] > 0, 1, 0)
    df['CarFlag_3'] = np.where(df['Car_3'] > 0, 1, 0)

    ## Read Distance Matrixes
    distmatrix = pd.read_csv('Dist_Matrix.csv')
    distmatrix.drop(columns='Unnamed: 0', inplace=True)

    ## Link Distances to Station1
    df = df.merge(distmatrix.set_index("Stations"), left_on="Station_1", right_index=True)

    ## Rename Columns again
    df.rename(columns={'LAT_x': 'LAT', 'LON_x': 'LON'}, inplace=True)

    ## Drop Unneeded columns
    df.drop(
        columns=['Car_1', 'Walk_1', 'Bus_1', 'Eng_Location', 'LAT_y', 'LON_y', 'Lat_Lon', 'Eng_Location_Eki1', 'Car_2',
                 'Walk_2', 'Bus_2', 'Eng_Location_Eki2', 'Car_3', 'Walk_3', 'Bus_3', 'Eng_Location_Eki3'], inplace=True)

    ## Clean up distance columns
    df.ShinjukuDist = df.ShinjukuDist.str.replace('km', '').str.strip()
    df.TokyoDist = df.TokyoDist.str.replace('km', '').str.strip()
    df.ShibuyaDist = df.ShibuyaDist.str.replace('km', '').str.strip()
    df.IkebukuroDist = df.IkebukuroDist.str.replace('km', '').str.strip()
    df.UenoDist = df.UenoDist.str.replace('km', '').str.strip()
    df.ShinagawaDist = df.ShinagawaDist.str.replace('km', '').str.strip()
    df.ShinjukuDist = df['ShinjukuDist'].astype('float32')
    df.TokyoDist = df['TokyoDist'].astype('float32')
    df.ShibuyaDist = df['ShibuyaDist'].astype('float32')
    df.IkebukuroDist = df['IkebukuroDist'].astype('float32')
    df.UenoDist = df['UenoDist'].astype('float32')
    df.ShinagawaDist = df['ShinagawaDist'].astype('float32')

    ## Crime

    ## Read Tokyo Crime data
    crime = pd.read_csv('Crime.csv')

    ## Clean crime data so it links with the original data
    crime['Location'] = crime['Location'].str.replace(' ', '')
    crime['Location'] = crime['Location'].str.replace('丁目', '')

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

    imputedf = pd.DataFrame(columns=crime.columns)
    imputedf['Location'] = np.array(['新宿区二十騎町', '千代田区西神田３', '新宿区市谷砂土原町３', '台東区谷中４', '杉並区成田西４', '北区堀船４',
                                     '千代田区神田東紺屋町', '新宿区市谷甲良町', '港区麻布狸穴町', '港区麻布永坂町', '新宿区市谷左内町'])
    imputeddf = imputedf.fillna(0)
    crime = pd.concat([crime, imputeddf], ignore_index=True)

    ## Merge Crime to base df
    df = df.merge(crime.set_index("Location"), left_on="location", right_index=True)

    ## Rename Total to Total Crimes for ease
    df.rename(columns={'Total': 'Total_Crimes'}, inplace=True)

    ## Environmental Data

    ### Air Quality

    ## Read Air Quality Data
    airquality = pd.read_csv('Air_Quality.csv')

    ## Clean Data so that it can be merged with main dataframe
    airquality['location'] = airquality.location.str.replace("東京都", "")

    ## Merge by Location
    df = df.merge(airquality.set_index("location"), left_on="location", right_index=True)

    ## Drop Unnecessary columns and rename
    df.drop(columns=['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor5', 'LAT_y', 'LON_y', 'Eng_Location'],
            inplace=True)
    df.rename(columns={'LAT_x': 'LAT', 'LON_x': 'LON'}, inplace=True)

    ### Bousai

    ## Read Earthquake Data
    bousai = pd.read_csv('Bousai.csv')

    ## Combine
    bousai['location'] = (bousai['Ku'].map(str) + bousai['location'].map(str))

    ## Replace Strings so that it can be joined
    bousai['location'] = bousai['location'].str.replace("　", "")
    bousai['location'] = bousai['location'].str.replace("丁目", "")

    ## Fix mismatches so that it can be joined
    ## 千代田区猿楽町１ >> 千代田区神田猿楽町２
    ## 千代田区猿楽町２ >> 千代田区神田猿楽町２
    ## 千代田区三崎町１ >> 千代田区神田三崎町１
    ## 千代田区三崎町２ >> 千代田区神田三崎町２
    ## 千代田区三崎町３ >> 千代田区神田三崎町３
    ## 新宿区本塩町 >> 新宿区四谷本塩町
    ## 新宿区三栄町 >> 新宿区四谷三栄町

    bousai['location'] = bousai['location'].str.replace("千代田区猿楽町１", "千代田区神田猿楽町１")
    bousai['location'] = bousai['location'].str.replace("千代田区猿楽町２", "千代田区神田猿楽町２")
    bousai['location'] = bousai['location'].str.replace("千代田区三崎町１", "千代田区神田三崎町１")
    bousai['location'] = bousai['location'].str.replace("千代田区三崎町２", "千代田区神田三崎町２")
    bousai['location'] = bousai['location'].str.replace("千代田区三崎町３", "千代田区神田三崎町３")
    bousai['location'] = bousai['location'].str.replace("新宿区本塩町", "新宿区四谷本塩町")
    bousai['location'] = bousai['location'].str.replace("新宿区三栄町", "新宿区四谷三栄町")

    ## Merge Data
    df = df.merge(bousai.set_index("location"), left_on="location", right_index=True)

    ## Rename Columns
    df.rename(columns={'Ku_x': 'Ku'}, inplace=True)
    df.drop(columns=['Ku_y'], inplace=True)

    # ## Parks

    ## Read Park Data
    parks = pd.read_csv('parkdata.csv')

    ## Merge Park Data to main df
    df = df.merge(parks.set_index("Ku"), left_on="Ku", right_index=True)

    ## Rename and drop columns
    df.rename(columns={'Area': 'Parks_Area', 'Total': 'Total_Parks'}, inplace=True)
    df.drop(columns=['Ku_japanese'], inplace=True)

    # ## Land Use

    ## Merge Land Use by District Data
    landusedistrict = pd.read_csv('LandUsebyDistrict.csv').drop(columns='Unnamed: 14')

    ## Drop Japanese Ku again
    landusedistrict.drop(columns='JapaneseKu', inplace=True)

    ## Merge to main DF
    df = df.merge(landusedistrict.set_index("Ku"), left_on="Ku", how='left', right_index=True)

    ## Rename ambigious columns
    landusedistrict.rename(columns={'Total': 'Total_LandUse'}, inplace=True)

    ## Rename columns in DF
    df.rename(columns={'Total': 'Total_District_Land'}, inplace=True)

    # ## Pollution Complaints

    ## Read Pollutions complaints data
    pollutioncomplaints = pd.read_csv('PollutionComplaints.csv').drop(index=[23, 24, 25, 26, 27, 28, 29],
                                                                      columns=['Unnamed: 11', 'Unnamed: 12',
                                                                               'Unnamed: 13', 'Unnamed: 14',
                                                                               'Unnamed: 15', 'Unnamed: 16',
                                                                               'Unnamed: 17',
                                                                               'Unnamed: 18', 'Unnamed: 19',
                                                                               'Unnamed: 20', 'Unnamed: 21',
                                                                               'Unnamed: 22', 'Unnamed: 23'])

    ## Drop Japanese KU
    pollutioncomplaints.drop(columns=['Japanese Ku'], inplace=True)

    ## Replace the dashes with 0's
    pollutioncomplaints['Water pollution'] = pollutioncomplaints['Water pollution'].replace('-', 0)
    pollutioncomplaints['Soil pollution'] = pollutioncomplaints['Soil pollution'].replace('-', 0)
    pollutioncomplaints['ground_subsidence'] = pollutioncomplaints['ground_subsidence'].replace('-', 0)
    pollutioncomplaints['other_pollution'] = pollutioncomplaints['other_pollution'].replace('-', 0)

    ## Merge Pollution Complains
    df = df.merge(pollutioncomplaints.set_index("Ku"), left_on="Ku", how='left', right_index=True)

    ## Drop Completely Empty features
    df.drop(columns=['Violent_Crime_Weapons', 'NO(ppm)_min', 'SPM(mg/m3)_min', 'WS(m/s)_min', 'ground_subsidence'],
            inplace=True)

    ## Fix Dtypes
    df['Population'] = df['Population'].str.replace(',', '').astype('int32')
    df['Population Density'] = df['Population Density'].str.replace(',', '').astype('float32')
    df['UP_Metropolitan_Parks_Area'] = df['UP_Metropolitan_Parks_Area'].str.replace(',', '').astype('float32')
    df['UP_Municipal_Parks_Area'] = df['UP_Municipal_Parks_Area'].str.replace(',', '').astype('float32')
    df['UP_National_Government_Parks_Area'] = df['UP_National_Government_Parks_Area'].str.replace(',', '').astype(
        'float32')
    df['UP_Area'] = df['UP_Area'].str.replace(',', '').astype('float32')
    df['Total_District_Land'] = df['Total_District_Land'].str.replace(',', '').astype('float32')
    df['Residentail'] = df['Residentail'].str.replace(',', '').astype('float32')
    df['Roads'] = df['Roads'].str.replace(',', '').astype('float32')
    df['roads2'] = df['roads2'].str.replace(',', '').astype('float32')
    df['Water pollution'] = df['Water pollution'].astype('float32')
    df['Soil pollution'] = df['Soil pollution'].astype('float32')
    df['other_pollution'] = df['other_pollution'].astype('float32')

    # ## Create Dataset GeoSpatial/Environmental Features Model
    print('------------CREATE GEO-ENV DATASETS---------------')
    X_train, X_valid, X_test, allcolumns, y_train, y_valid, y_test = pre_process(df, ['title', 'rent'])

    ## Save Training, Valid, and Test Files
    print('------------SAVE GEO-ENV DATASETS---------------')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/train')

    sparse.save_npz('X_train.npz', X_train)
    y_train.to_hdf('y_train.hdf', 'train')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/valid')
    sparse.save_npz('X_valid.npz', X_valid)
    y_valid.to_hdf('y_valid.hdf', 'valid')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/test')
    sparse.save_npz('X_test.npz', X_test)
    y_test.to_hdf('y_test.hdf', 'test')

    ## Columns for LGBM
    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/geo_env_model/')
    pd.DataFrame(allcolumns).to_csv('columns.csv', index=False)

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

    ## Create Category for Most common layout in region
    mostcommonlayout = df.groupby(['location'])['floor_plan'].apply(lambda x: x.mode()).reset_index()
    mostcommonlayout.rename(columns={'floor_plan': 'floor_plan_most_common_FE'}, inplace=True)
    ## For now take level 0 - Maybe convert anything with level 1 to a multiple class for the algorithm
    mostcommonlayout = mostcommonlayout[mostcommonlayout['level_1'] == 0]
    mostcommonlayout.drop(columns=['level_1'], inplace=True)

    df = df.merge(mostcommonlayout.set_index('location'), left_on='location', right_index=True)

    # ##### Expansion Encoding

    ## 5/14 Create Room Features for Living, Dining, Kitchen, Service_Room, and 1RM -
    df['Living_FE'] = np.where(df['floor_plan'].str.contains('L') == True, 1, 0)
    df['Dining_FE'] = np.where(df['floor_plan'].str.contains('D') == True, 1, 0)
    df['Kitchen_FE'] = np.where(df['floor_plan'].str.contains('K') == True, 1, 0)
    df['Service_Room_FE'] = np.where(df['floor_plan'].str.contains('S') == True, 1, 0)
    df['Room_Only_FE'] = np.where(df['floor_plan'].str.contains('RM') == True, 1, 0)

    ## Remove Strings from floor_plan
    df['floor_plan'] = df['floor_plan'].str.replace('L', '')
    df['floor_plan'] = df['floor_plan'].str.replace('D', '')
    df['floor_plan'] = df['floor_plan'].str.replace('K', '')
    df['floor_plan'] = df['floor_plan'].str.replace('S', '')
    df['floor_plan'] = df['floor_plan'].str.replace('RM', '')

    ## rename floor_plan to total rooms
    df.rename(columns={'floor_plan': 'total_rooms_FE'}, inplace=True)

    ## Convert total_rooms to int
    df['total_rooms_FE'] = df['total_rooms_FE'].astype('int')

    # #### Interaction Features

    ## Area per Room ## INTERACTION
    df['area_per_room_FE'] = df['area'] / df['total_rooms_FE']

    ## 5/15 ## INTERACTION
    ## Ratio of Floor Living on compared to total building size
    df['height_ratio_FE'] = df['floor'].astype(int) / df['Height']

    ## Interaction Features - Max Corrs - Highest Correlated Features Multiplied
    df['area_height_FE'] = df['area'] * df['Height']

    ## If Living on First Floor
    df['first_floor_FE'] = np.where(df['floor'] == 1, 1, 0)

    ## Interaction feature-  floor vs area
    df['floor_area_FE'] = df['floor'].astype(int) * df['area']

    ## 5/17  - Unfortunately Building material Data doesnt exist so building risk will need to be modeling in other ways

    ## Ideas from this paper http://hermes-ir.lib.hit-u.ac.jp/rs/bitstream/10086/13412/1/0100701601.pdf
    ## http://japanpropertycentral.com/real-estate-faq/earthquake-building-codes-in-japan/

    ## Interaction between Age of Building / Risk Rating
    df['yrs_risk_FE'] = df['yrs'] * df['Total_Risk']

    ## Interaction between Age of Building / LifeDifficulty Rating
    df['yrs_lifediff_FE'] = df['yrs'] * df['Life_Difficulty_Risk']
    ## Interaction between Age of Building / Fire Rating
    df['yrs_fire_FE'] = df['yrs'] * df['Fire_Risk']

    ## Interaction between Age of Building / Building Rating
    df['yrs_building_FE'] = df['yrs'] * df['Building_Risk']

    # ##### Binned Features

    ### Done ####
    ## Group Minority Administrative Districts into 1 classification
    ## Minority class bins

    locationgroup = df.groupby(['Ku', 'location'])['title'].count()
    locationgroup = locationgroup.reset_index()
    locationgroupminority = locationgroup[locationgroup.title < 5]

    for i in locationgroupminority.Ku.unique():
        group = locationgroupminority['location'][locationgroupminority.Ku == i]
        df['location'] = np.where(df['location'].isin(group), str(i + '_Minority'), df['location'])

    ## Group Minority Lines Together into 1 classification
    linegroup = df.groupby(['Line_1'])['title'].count()
    linegroup = linegroup.reset_index()
    linegroupminority = linegroup[linegroup.title < 30]
    df['Line_1'] = np.where(df['Line_1'].isin(linegroupminority['Line_1']), str('Line_Minority'), df['Line_1'])

    ## Group Minority Stations Together into 1 classification
    stationgroup = df.groupby(['Station_1'])['title'].count()
    stationgroup = stationgroup.reset_index()
    stationgroupminority = stationgroup[stationgroup.title < 30]
    df['Station_1'] = np.where(df['Station_1'].isin(stationgroupminority['Station_1']), str('Station_Minority'),
                               df['Station_1'])

    ## Group Minority Line 2 Together into 1 classification
    linegroup2 = df.groupby(['Line_2'])['title'].count()
    linegroup2 = linegroup2.reset_index()
    linegroupminority2 = linegroup2[linegroup2.title < 30]
    df['Line_2'] = np.where(df['Line_2'].isin(linegroupminority2['Line_2']), str('Line_Minority'), df['Line_2'])

    ## Group Minority Station 2 Together into 1 classification
    stationgroup2 = df.groupby(['Station_2'])['title'].count()
    stationgroup2 = stationgroup2.reset_index()
    stationgroupminority2 = stationgroup2[stationgroup2.title < 30]
    df['Station_2'] = np.where(df['Station_2'].isin(stationgroupminority2['Station_2']), str('Station_Minority'),
                               df['Station_2'])

    ## Group Minority Line 3 Together into 1 classification
    linegroup3 = df.groupby(['Line_3'])['title'].count()
    linegroup3 = linegroup3.reset_index()
    linegroupminority3 = linegroup3[linegroup3.title < 30]
    df['Line_3'] = np.where(df['Line_3'].isin(linegroupminority3['Line_3']), str('Line_Minority'), df['Line_3'])

    ## Group Minority Station 3 Together into 1 classification
    stationgroup3 = df.groupby(['Station_3'])['title'].count()
    stationgroup3 = stationgroup3.reset_index()
    stationgroupminority3 = stationgroup3[stationgroup3.title < 30].sort_values(by='title')
    df['Station_3'] = np.where(df['Station_3'].isin(stationgroupminority3['Station_3']), str('Station_Minority'),
                               df['Station_3'])

    # #### Transit Interactions

    ## Total Time for All Hubs - Added together
    df['Hub_Aggregate_FE'] = df['ShinjukuMade'] + df['TokyoMade'] + df['ShibuyaMade'] + df['IkebukuroMade'] + df[
        'UenoMade'] + df['ShinagawaMade']

    ## Calculate Commute Time - Walk + Transit ## INTERACTION FEATURES
    df['Shinjuku_Commute_FE'] = df['Time_to_Station1'] + df['ShinjukuMade']
    df['Tokyo_Commute_FE'] = df['Time_to_Station1'] + df['TokyoMade']
    df['Shibuya_Commute_FE'] = df['Time_to_Station1'] + df['ShibuyaMade']
    df['Ikebukuro_Commute_FE'] = df['Time_to_Station1'] + df['IkebukuroMade']
    df['Ueno_Commute_FE'] = df['Time_to_Station1'] + df['UenoMade']
    df['Shinagawa_Commute_FE'] = df['Time_to_Station1'] + df['ShinagawaMade']

    ## Sum of all Commute Times INTERACTION FEATURE
    df['Aggregate_Commute_FE'] = df['Shinjuku_Commute_FE'] + df['Tokyo_Commute_FE'] + df['Shibuya_Commute_FE'] + df[
        'Ikebukuro_Commute_FE'] + df['Ueno_Commute_FE'] + df['Shinagawa_Commute_FE']

    # #### More Interactions
    ## Interacts between aggregate AGG_Commute and building/disaster related attributes -- INTERACTION FEATURES
    df['agg_commute_floor_area_FE'] = df['Aggregate_Commute_FE'] * df['floor_area_FE']
    df['agg_commute_area_height_FE'] = df['Aggregate_Commute_FE'] * df['area_height_FE']
    df['agg_commute_area_per_room_FE'] = df['Aggregate_Commute_FE'] * df['area_per_room_FE']
    df['agg_commute_yrs_risk_FE'] = df['Aggregate_Commute_FE'] * df['yrs_risk_FE']
    df['agg_commute_yrs_lifediff_FE'] = df['Aggregate_Commute_FE'] * df['yrs_lifediff_FE']
    df['agg_commute_yrs_fire_FE'] = df['Aggregate_Commute_FE'] * df['yrs_fire_FE']
    df['agg_commute_yrs_building_FE'] = df['Aggregate_Commute_FE'] * df['yrs_building_FE']

    # #### Consolidation Encoding

    ### Done ####
    ## Create Categorical Features for District of Tokyo - EXPANSION ENCODING ?
    df['Region'] = np.where(df['Ku'].isin(['Itabashi', 'Kita']), 'North_outer', None)
    df['Region'] = np.where(df['Ku'].isin(['Nerima', 'Suginami', 'Nakano']), 'West', df['Region'])
    df['Region'] = np.where(df['Ku'].isin(['Toshima', 'Bunkyo']), 'North_inner', df['Region'])
    df['Region'] = np.where(df['Ku'].isin(['Arakawa', 'Adachi', 'Katsushika', 'Edogawa']), 'East_outer', df['Region'])
    df['Region'] = np.where(df['Ku'].isin(['Koto', 'Taito', 'Sumida']), 'East_inner', df['Region'])
    df['Region'] = np.where(df['Ku'].isin(['Shibuya', 'Shinjuku', 'Minato', 'Chiyoda', 'Chuo']), 'Central',
                            df['Region'])
    df['Region'] = np.where(df['Ku'].isin(['Setagaya', 'Meguro', 'Shinagawa', 'Ota']), 'South', df['Region'])

    ######## CRIME #########
    ##  Ranking / INTERACTION FEATURES

    ## Python create overall Crime ranks
    df['Crime_Rank_FE'] = df['Total_Crimes'].rank(method='dense', ascending=False)

    ## Felonies
    df['Felony_Offense_Ratio_FE'] = df['Felonious_Offense_Total'] / df['Total_Crimes']
    df['Felony_Rank_FE'] = df['Felonious_Offense_Total'].rank(method='dense', ascending=False)
    ## Violent Crime Rate - Violent Crime Rank
    df['Violent_Crime_Ratio_FE'] = df['Violent_Crime_Total'] / df['Total_Crimes']
    df['Violent_Crime_Rank_FE'] = df['Violent_Crime_Total'].rank(method='dense', ascending=False)

    ## Burglary Larceny Rate - Burglary Larceny Rank
    df['Burglary_Larceny_Ratio_FE'] = df['Burglary_Larceny_Total'] / df['Total_Crimes']
    df['Burglary_Larceny_Rank_FE'] = df['Burglary_Larceny_Total'].rank(method='dense', ascending=False)

    ## Non_Intrusive Larceny Rate - Non_Intrusive_Larceny_Rank
    df['Non_Intrusive_Larceny_Ratio_FE'] = df['Non_Intrusive_Larceny'] / df['Total_Crimes']
    df['Non_Intrusive_Larceny_Rank_FE'] = df['Non_Intrusive_Larceny'].rank(method='dense', ascending=False)

    ## Others Rate - Others Rank
    df['Other_Crime_Ratio_FE'] = df['Others_Total'] / df['Total_Crimes']
    df['Other_Crime_Rank_FE'] = df['Others_Total'].rank(method='dense', ascending=False)

    ######### LAND USE ########
    ## INTERACTION FEATURES

    ## Land Use 2 - Calculate Percentage of Land by Ku
    df['Residentail_land_ratio_FE'] = df['Residentail'] / df['Total_District_Land']
    df['OtherUse_land_ratio_FE'] = df['OtherUse'] / df['Total_District_Land']
    df['Parks_land_ratio_FE'] = df['Parks'] / df['Total_District_Land']
    df['Unused_land_ratio_FE'] = df['Unused'] / df['Total_District_Land']
    df['Roads_land_ratio_FE'] = df['Roads'] / df['Total_District_Land']
    df['Farm_land_ratio_FE'] = df['Farm'] / df['Total_District_Land']
    df['Water_land_ratio_FE'] = df['Water'] / df['Total_District_Land']
    df['WoodsForest_land_ratio_FE'] = df['WoodsForest'] / df['Total_District_Land']
    df['Fields_land_ratio_FE'] = df['Fields'] / df['Total_District_Land']

    ######## POLLUTION COMPLAINTS ###########
    ## INTERACTION FEATURES

    ## Pollution Complaints - Calculate Pollution complaints by population
    df['Air_Pollution_Complaints_per_Person_FE'] = df['Air pollution'] / df['Population']
    df['Water_Pollution_Complaints_per_Person_FE'] = df['Water pollution'] / df['Population']
    df['Soil_Pollution_Complaints_per_Person_FE'] = df['Soil pollution'] / df['Population']
    df['Noise_Pollution_Complaints_per_Person_FE'] = df['Noise'] / df['Population']
    df['Vibration_Complaints_per_Person_FE'] = df['Vibration'] / df['Population']
    df['Offensive_odors_Complaints_per_Person_FE'] = df['Offensive_odors'] / df['Population']
    df['Other_Pollution_Complaints_per_Person_FE'] = df['other_pollution'] / df['Population']

    #### Cartesian Projections ####
    ## Cartesian Projection
    df['x_cart_FE'] = np.cos(df['LAT']) * np.cos(df['LON'])
    df['y_cart_FE'] = np.cos(df['LAT']) * np.sin(df['LON'])
    df['z_cart_FE'] = np.sin(df['LAT'])

    ## Cartesian Projection Station 1
    df['x_cart_Eki1_FE'] = np.cos(df['LAT_Eki1']) * np.cos(df['LON_Eki1'])
    df['y_cart_Eki1_FE'] = np.cos(df['LAT_Eki1']) * np.sin(df['LON_Eki1'])
    df['z_cart_Eki1_FE'] = np.sin(df['LAT_Eki1'])

    ## Cartesian Projection Station 2
    df['x_cart_Eki2_FE'] = np.cos(df['LAT_Eki2']) * np.cos(df['LON_Eki2'])
    df['y_cart_Eki2_FE'] = np.cos(df['LAT_Eki2']) * np.sin(df['LON_Eki2'])
    df['z_cart_Eki2_FE'] = np.sin(df['LAT_Eki2'])

    ## Cartesian Projection Station 3
    df['x_cart_Eki3_FE'] = np.cos(df['LAT_Eki3']) * np.cos(df['LON_Eki3'])
    df['y_cart_Eki3_FE'] = np.cos(df['LAT_Eki3']) * np.sin(df['LON_Eki3'])
    df['z_cart_Eki3_FE'] = np.sin(df['LAT_Eki3'])

    # ### Target Encoding

    ## TARGET ENCODING
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

    categoricalcolumns = X_train.select_dtypes(['object', 'category']).drop(
        columns=['title', 'Ku', 'Land Classification', 'Region', 'floor_plan_most_common_FE']).columns

    ## Calculate cutoff values for target encoding

    ## 10 Location
    print(np.percentile(X_train.groupby('location').agg('count')['title'].sort_values(), 20), 'percentile for counts')

    ## 85 - Line 1
    print(np.percentile(X_train.groupby('Line_1').agg('count')['title'].sort_values(), 20), 'percentile for counts')
    ## 55 - Station 1
    print(np.percentile(X_train.groupby('Station_1').agg('count')['title'].sort_values(), 20), 'percentile for counts')
    ## 125 - Line_2
    print(np.percentile(X_train.groupby('Line_2').agg('count')['title'].sort_values(), 20), 'percentile for counts')
    ## 50 - Station_2
    print(np.percentile(X_train.groupby('Station_2').agg('count')['title'].sort_values(), 20), 'percentile for counts')
    ## 50 - Line_3
    print(np.percentile(X_train.groupby('Line_3').agg('count')['title'].sort_values(), 20), 'percentile for counts')
    ## 45 - Station_3
    print(np.percentile(X_train.groupby('Station_3').agg('count')['title'].sort_values(), 20), 'percentile for counts')

    samples = [15, 152, 87, 300, 92, 278, 78]

    for category, sample in zip(categoricalcolumns, samples):
        # print('Min Samples for Leaf-',sample)
        X_train[category], X_valid[category], X_test[category] = target_encode(train=X_train[category],
                                                                               valid=X_valid[category],
                                                                               test=X_test[category],
                                                                               target=y_train,
                                                                               min_samples_leaf=sample,
                                                                               smoothing=10,
                                                                               noise_level=.001)
        print('Target Encoding-', category)

    target = pd.concat([y_train, y_valid, y_test])

    df = pd.concat([X_train, X_valid, X_test])

    df['target'] = target

    df.rename(columns={'location': 'location_FE',
                       'Line_1': 'Line_1_FE',
                       'Station_1': 'Station_1_FE',
                       'Line_2': 'Line_2_FE',
                       'Station_2': 'Station_2_FE',
                       'Line_3': 'Line_3_FE',
                       'Station_3': 'Station_3_FE'}, inplace=True)

    ## Return Correlations
    print(df.corr()['target'].sort_values())

    print('------------CREATE FEATURE ENGINEERING DATASETS---------------')

    X_train, X_valid, X_test, allcolumns, y_train, y_valid, y_test = pre_process(df, ['title', 'rent'])

    ## Create Data - Feature Engineering Model
    print('------------SAVE FEATURE ENGINEERING DATASETS---------------')

    ## Save Training, Valid, and Test Files
    ## Save Training, Valid, and Test Files
    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/train')

    sparse.save_npz('X_train.npz', X_train)
    y_train.to_hdf('y_train.hdf', 'train')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/valid')
    sparse.save_npz('X_valid.npz', X_valid)
    y_valid.to_hdf('y_valid.hdf', 'valid')

    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/test')
    sparse.save_npz('X_test.npz', X_test)
    y_test.to_hdf('y_test.hdf', 'test')

    ## Columns for LGBM
    os.chdir('/Volumes/SeagateExternal/Masters_Thesis_Tokyo_Rent_Prediction/data/FE_model/')
    pd.DataFrame(allcolumns).to_csv('columns_FE.csv', index=False)
