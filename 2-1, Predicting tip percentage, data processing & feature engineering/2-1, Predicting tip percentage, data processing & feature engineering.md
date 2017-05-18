
## This notebook is for exploratory data analysis, cleaning and feature engineering for Task 4. Processed data will be saved as pickle files for modeling and quick recovery from unexpected system failure

## Task 4
* 4.1 Build a derived variable for tip as a percentage of the total fare.
* 4.2 Build a predictive model for tip as a percentage of the total fare. Use as much of the data as you like (or all of it). We will validate a sample.

Before jumping into any anlysis, it's helpful to use some intuition to think about what features can be used or engineered to predict the percentage of the total fare, here is a list that I came up with:

<table>
    <tr>
        <th>Feature Name</th>
        <th>Type</th>
        <th>Note</th>
    </tr>
    
    <tr>
        <td>vendor_id</td>
        <td>categorical</td>
        <td></td>
    </tr>
    
    <tr>
        <td>duration</td>
        <td>numerical, continuous</td>
        <td>dropoff time - pickup time</td>
    </tr>
    
    <tr>
        <td>pickup_hr</td>
        <td>categorical</td>
        <td>the hour of the day when pickup occurred. Value ranges from 0 to 23</td>
    </tr>
    
    <tr>
        <td>pickup_boro</td>
        <td>categorical</td>
        <td>which borough did the pickup occur. Value ranges from 1 to 5, 0 denotes none of the five areas.</td>
    </tr>
    
    <tr>
        <td>drop_boro</td>
        <td>categorical</td>
        <td>similar to pickup_boro</td>
    </tr>
    
    <tr>
        <td>rate_code</td>
        <td>categorical</td>
        <td></td>
    </tr>
    
    <tr>
        <td>trip_distance</td>
        <td>numerical, continuous</td>
        <td></td>
    </tr>
    
    <tr>
        <td>store_and_fwd_flag</td>
        <td>categorical</td>
        <td>true/false</td>
    </tr>
    
    <tr>
        <td>payment_type</td>
        <td>categorical</td>
        <td></td>
    </tr>
    
    <tr>
        <td>total_amount</td>
        <td>numerical, continuous</td>
        <td>fare_amount in the dataset represents payment calculated by distance while total_amount is the total payment incurred by a trip without tip. We will use total_amount instead of fare_amount because tip decisions are often made based upon total amount.</td>
    </tr>
    
    <tr>
        <td>passenger_count</td>
        <td>numerical, discrete</td>
        <td></td>
    </tr>
    
    <tr>
        <td>trip_type</td>
        <td>categorical</td>
        <td></td>
    </tr>
</table>

## magic shortcuts to subsections:

* [Explorary data analysis](#1)

* [Data cleaning & Fearture engineering](#2)


```python
# load library
import pandas as pd # data analysis
pd.options.display.float_format = '{:,.3f}'.format
import seaborn as sns # visualization
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import geopandas as gpd # for processing geographical varibales
from geopandas import GeoSeries, GeoDataFrame
from shapely.geometry import Point, Polygon, LineString
import pickle 
```


```python
# load data
dataset = pd.read_csv('./data/green_tripdata_2015-09.csv')
```

<a id = 1> </a>
### Explorary data analysis
Before diving into more details, we will do some explorary data analysis to develop a better undertanding of the dataset:

* <b> missing values </b>


```python
# print out the number of missing values associated with each feature
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1494926 entries, 0 to 1494925
    Data columns (total 21 columns):
    VendorID                 1494926 non-null int64
    lpep_pickup_datetime     1494926 non-null object
    Lpep_dropoff_datetime    1494926 non-null object
    Store_and_fwd_flag       1494926 non-null object
    RateCodeID               1494926 non-null int64
    Pickup_longitude         1494926 non-null float64
    Pickup_latitude          1494926 non-null float64
    Dropoff_longitude        1494926 non-null float64
    Dropoff_latitude         1494926 non-null float64
    Passenger_count          1494926 non-null int64
    Trip_distance            1494926 non-null float64
    Fare_amount              1494926 non-null float64
    Extra                    1494926 non-null float64
    MTA_tax                  1494926 non-null float64
    Tip_amount               1494926 non-null float64
    Tolls_amount             1494926 non-null float64
    Ehail_fee                0 non-null float64
    improvement_surcharge    1494926 non-null float64
    Total_amount             1494926 non-null float64
    Payment_type             1494926 non-null int64
    Trip_type                1494922 non-null float64
    dtypes: float64(14), int64(4), object(3)
    memory usage: 239.5+ MB


Ehail_fee is completely missing, we will impute this feature. Other than this feature, Trip_type has only four missing values, we will ignore these cases.


```python
# impute Ehail_fee because all of them are 'NA'
dataset.drop('Ehail_fee', axis = 1, inplace = True)
```

* <b> non-sense values </b>

To investigate non-sense values, we will first take a look at a summary statistics of all numerical values:


```python
dataset.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>RateCodeID</th>
      <th>Pickup_longitude</th>
      <th>Pickup_latitude</th>
      <th>Dropoff_longitude</th>
      <th>Dropoff_latitude</th>
      <th>Passenger_count</th>
      <th>Trip_distance</th>
      <th>Fare_amount</th>
      <th>Extra</th>
      <th>MTA_tax</th>
      <th>Tip_amount</th>
      <th>Tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>Total_amount</th>
      <th>Payment_type</th>
      <th>Trip_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,926.00000</td>
      <td>1,494,922.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.78204</td>
      <td>1.09765</td>
      <td>-73.83084</td>
      <td>40.69114</td>
      <td>-73.83728</td>
      <td>40.69291</td>
      <td>1.37060</td>
      <td>2.96814</td>
      <td>12.54320</td>
      <td>0.35128</td>
      <td>0.48664</td>
      <td>1.23573</td>
      <td>0.12310</td>
      <td>0.29210</td>
      <td>15.03215</td>
      <td>1.54056</td>
      <td>1.02235</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.41286</td>
      <td>0.63594</td>
      <td>2.77608</td>
      <td>1.53088</td>
      <td>2.67791</td>
      <td>1.47670</td>
      <td>1.03943</td>
      <td>3.07662</td>
      <td>10.08278</td>
      <td>0.36631</td>
      <td>0.08504</td>
      <td>2.43148</td>
      <td>0.89101</td>
      <td>0.05074</td>
      <td>11.55316</td>
      <td>0.52329</td>
      <td>0.14783</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-83.31908</td>
      <td>0.00000</td>
      <td>-83.42784</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>-475.00000</td>
      <td>-1.00000</td>
      <td>-0.50000</td>
      <td>-50.00000</td>
      <td>-15.29000</td>
      <td>-0.30000</td>
      <td>-475.00000</td>
      <td>1.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.00000</td>
      <td>1.00000</td>
      <td>-73.95961</td>
      <td>40.69895</td>
      <td>-73.96782</td>
      <td>40.69878</td>
      <td>1.00000</td>
      <td>1.10000</td>
      <td>6.50000</td>
      <td>0.00000</td>
      <td>0.50000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.30000</td>
      <td>8.16000</td>
      <td>1.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.00000</td>
      <td>1.00000</td>
      <td>-73.94536</td>
      <td>40.74674</td>
      <td>-73.94504</td>
      <td>40.74728</td>
      <td>1.00000</td>
      <td>1.98000</td>
      <td>9.50000</td>
      <td>0.50000</td>
      <td>0.50000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.30000</td>
      <td>11.76000</td>
      <td>2.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.00000</td>
      <td>1.00000</td>
      <td>-73.91748</td>
      <td>40.80255</td>
      <td>-73.91013</td>
      <td>40.79015</td>
      <td>1.00000</td>
      <td>3.74000</td>
      <td>15.50000</td>
      <td>0.50000</td>
      <td>0.50000</td>
      <td>2.00000</td>
      <td>0.00000</td>
      <td>0.30000</td>
      <td>18.30000</td>
      <td>2.00000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.00000</td>
      <td>99.00000</td>
      <td>0.00000</td>
      <td>43.17726</td>
      <td>0.00000</td>
      <td>42.79934</td>
      <td>9.00000</td>
      <td>603.10000</td>
      <td>580.50000</td>
      <td>12.00000</td>
      <td>0.50000</td>
      <td>300.00000</td>
      <td>95.75000</td>
      <td>0.30000</td>
      <td>581.30000</td>
      <td>5.00000</td>
      <td>2.00000</td>
    </tr>
  </tbody>
</table>
</div>



It's easy to spot some non-sense values from the table above. For example, Tip_amount, Tolls_amount, MTA_tax can't be negative. Trip_distnace can't be 0 mile. We will keep these observations in mind and impute these non-sense values later.

* <b> geographical locations </b>

Next, we will plot pickup and dropoff locations on NYC map to further investigate data quality. This analysis was inspired by this article: http://blog.yhat.com/posts/interactive-geospatial-analysis.html


```python
# load NYC map, downloaded from 
# http://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/nybb/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=geojson
nyc_map = gpd.read_file('./data/query.txt')
```


```python
# visualize pick up locations
# we will plot each pick up location as a redish point on the map
plt.style.use('bmh')

# set figure size
plt.rcParams['figure.figsize'] = (16.0, 16.0)

fig, ax = plt.subplots()

ax.set_aspect('equal')

base = nyc_map.plot(ax = ax)

# plot locations on the map
plt.scatter(x=dataset['Pickup_longitude'], y=dataset['Pickup_latitude'], alpha=0.1, c='r')
```




    <matplotlib.collections.PathCollection at 0x118bc6f60>




![png](output_15_1.png)


The plot above shows that there exists pickup locations apparently outside of NYC. We will next show only locations with longitude from -73.6 to -74.3 and latitude from 40.4 to 41, which is at roughly the same area as NYC is.


```python
plt.style.use('bmh')

# set figure size
plt.rcParams['figure.figsize'] = (16.0, 16.0)

fig, ax = plt.subplots()

ax.set_aspect('equal')

base = nyc_map.plot(ax = ax)

# set filter criteria to only show points within NYC area
idx = ((dataset['Pickup_longitude'] > -74.3)  & (dataset['Pickup_longitude'] < -73.6) & 
(dataset['Pickup_latitude'] > 40.4) & (dataset['Pickup_latitude'] < 41))

# plot locations on the map
plt.scatter(x=dataset['Pickup_longitude'][idx], y=dataset['Pickup_latitude'][idx], alpha=0.1, c='r')
```




    <matplotlib.collections.PathCollection at 0x119dde940>




![png](output_17_1.png)


The pickup location data starts to make sense after applying this filtering criterion. The same situation applies to dropoff location data.

<a id = 2> </a>
### Data cleaning & Freature engineering

Create a new feature to represent the duration of each trip in minutes


```python
# convert to datetime format
dataset['lpep_pickup_datetime'] = pd.to_datetime(dataset['lpep_pickup_datetime'])
dataset['Lpep_dropoff_datetime'] = pd.to_datetime(dataset['Lpep_dropoff_datetime'])

# calculate duration
duration = dataset['Lpep_dropoff_datetime'] - dataset['lpep_pickup_datetime']

# convert to minutes in numerical format
duration = pd.DatetimeIndex(duration)
duration = duration.hour * 60 + duration.minute 
```

Create a new feature to represent the hour of the day when the trip started


```python
# convert to datetime format
pick_up_hr = pd.DatetimeIndex(dataset['lpep_pickup_datetime'])
# get hour of the day
pick_up_hr = pick_up_hr.hour
```

Create a new feature to represent the area of pickup and dropoff locations by mapping coordinates into the five borough areas in NYC


```python
# Mapping each coordinate data to NYC boroughs is a computationally expensive job. 
# We'll therefore relax the NYC map into polygons to approximate the five boroughs.
nyc_hull = nyc_map['geometry'].convex_hull
```


```python
# define a function that takes location coordinates as input and produce borough code as output
def whichBorough(point):
    '''
    return 1-5 if the point is mapped into NYC boroughs
    return 0 otherwise
    ---------
    args: 
        point, coordinate values, eg. (40.773460, -73.890349)
    ---------
    Five boroughs in NYC:
    1, Manhattan
    2, Bronx
    3, Brooklyn
    4, Queens
    5, Staten Island
        -- https://en.wikipedia.org/wiki/List_of_counties_in_New_York
    '''
    # conver data type
    point = Point(point)
    # determine which boro has been mapped into
    which_boro = nyc_hull.contains(point).values
    # if position can not be mapped into one of the five boros, reutrn 0
    if which_boro.sum() == 0:
        return 0
    # if position mapped to one of the five boros, return corresponding boro code
    else:
        return which_boro.argmax() + 1  
```


```python
# this is a computationally expensive job, we will save results for future reference
# pickup boros
pickup_boro = dataset[['Pickup_longitude', 'Pickup_latitude']].apply(lambda x: whichBorough(x), axis = 1)
# dropoff boros
dropoff_boro = dataset[['Dropoff_longitude', 'Dropoff_latitude']].apply(lambda x: whichBorough(x), axis = 1)
```


```python
## save pickup_boro and dropoff_boro for future reference
with open('./data/pickup_boro.p', 'wb') as handle:
    pickle.dump(pickup_boro, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('./data/dropoff_boro.p', 'wb') as handle:
    pickle.dump(dropoff_boro, handle, protocol=pickle.HIGHEST_PROTOCOL)
```


```python
# next time, load from pickle
pickup_boro = pickle.load(open('./data/pickup_boro.p', 'rb' ))

dropoff_boro = pickle.load(open('./data/dropoff_boro.p', 'rb' ))
```


```python
# compile features together
dataset2 = pd.DataFrame({'vendor_id': dataset['VendorID'], 'duration' : duration, 'pick_up_hr' : pick_up_hr,
                        'pickup_boro': pickup_boro, 'dropoff_boro': dropoff_boro, 'rate_code': dataset['RateCodeID'], 
                        'trip_distance': dataset['Trip_distance'], 'store_and_fwd_flag': dataset['Store_and_fwd_flag'],
                        'payment_type': dataset['Payment_type'], 'total_amount': dataset['Total_amount'], 'passenger_count': dataset['Passenger_count'],
                        'trip_type': dataset['Trip_type '], 'tip_amount': dataset['Tip_amount'],
                        'fare_amount': dataset['Fare_amount']})
```

data cleaning on features of 'durations', 'total_amount', 'trip_amount' and 'fare_amount' by excluding non-positive observations


```python
dataset3 = dataset2[(dataset2['duration'] > 0) & (dataset2['total_amount'] > 0) &
                    (dataset2['trip_distance'] > 0) & (dataset2['fare_amount'] > 0)]
```

let's look at the distribution of 'durations', 'total_amount', 'trip_amount' and 'fare_amount' 


```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['duration'], axlabel = 'duration')
axes = hist.axes
```


![png](output_34_0.png)



```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['total_amount'], axlabel = 'total_amount')
axes = hist.axes
```


![png](output_35_0.png)



```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['trip_distance'], axlabel = 'trip distance')
axes = hist.axes
```


![png](output_36_0.png)



```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['fare_amount'], axlabel = 'fare amount')
axes = hist.axes
```


![png](output_37_0.png)


all of the four features are long-tailed distributed, we will further clean them up


```python
dataset3 = dataset3[dataset3['duration'] < 90]
dataset3 = dataset3[dataset3['total_amount'] < 100]
dataset3 = dataset3[dataset3['trip_distance'] < 100]
dataset3 = dataset3[dataset3['fare_amount'] < 100]
```

distribution plot after data cleaning makes more sense:


```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['duration'], axlabel = 'duration')
axes = hist.axes
```


![png](output_41_0.png)



```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['total_amount'], axlabel = 'total_amount')
axes = hist.axes
```


![png](output_42_0.png)



```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['trip_distance'], axlabel = 'trip distance')
axes = hist.axes
```


![png](output_43_0.png)



```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['fare_amount'], axlabel = 'fare amount')
axes = hist.axes
```


![png](output_44_0.png)


data clearning to exclude trip that happened outside NYC


```python
# remove trip with neither start and end point outside of the five boroughs in NYC
dataset3 = dataset3[(dataset3['pickup_boro'] != 0) | (dataset3['dropoff_boro'] != 0)]
```

Create a new feature to represent the average speed miles/hr


```python
# average speed, miles/hr
dataset3['avg_speed'] = dataset3['trip_distance'] / (dataset3['duration'] / 60)
```


```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['avg_speed'], axlabel = 'avg speed (miles/hr)')
axes = hist.axes
```


![png](output_49_0.png)


data cleaning to exclude observations with non-sense speed (> 60 miles/hr)


```python
dataset3 = dataset3[dataset3['avg_speed'] < 60]
```


```python
# distribution plot afterwards
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['avg_speed'], axlabel = 'avg speed (miles/hr)')
axes = hist.axes
```


![png](output_52_0.png)


calculate the percentage of tips in terms of the total fare


```python
dataset3['tip_percent'] = dataset3['tip_amount'] / dataset3['fare_amount']
```

let's get some summary statistics first:


```python
dataset3['tip_percent'].describe()
```




    count   1,447,018.000
    mean            0.093
    std             0.178
    min             0.000
    25%             0.000
    50%             0.000
    75%             0.214
    max            99.000
    Name: tip_percent, dtype: float64



It seems over half of transactions didn't have a tip at all. The maximum percentage is 99%, which doesn't make sense. We will only include trips with tip percentage greater than 0% and less than 50% in our modeling


```python
dataset3 = dataset3[(dataset3['tip_percent'] > 0) & (dataset3['tip_percent'] < .50)]
```

distribution plot after further cleaning


```python
sns.set(rc={"figure.figsize": (8, 4)});
hist = sns.distplot(dataset3['tip_percent'], axlabel = 'tip %', bins = 100)
axes = hist.axes
```


![png](output_60_0.png)


Up to now, the data is almost ready to use. We will further apply several transformations:

* One-hot encoding on categorical features:


```python
# transform string-valued feature to integer-valued feature
dataset3['store_and_fwd_flag'].replace(['N', 'Y'], [0, 1], inplace = True)

# categorical feature list
cat_cols = ['pickup_boro', 'dropoff_boro', 'payment_type', 
            'store_and_fwd_flag', 'trip_type', 'pick_up_hr', 
            'vendor_id', 'rate_code', 'passenger_count']

# import one-hot encoding package
from sklearn import preprocessing

# apply one-hot endoing and transform into sparse matrix
OHE = preprocessing.OneHotEncoder(sparse=True)
dataset3_ohe = OHE.fit_transform(dataset3[cat_cols])
```


```python
# numerical values that will be included in the modeling
num_cols = ['duration', 'fare_amount', 'total_amount', 'trip_distance', 'avg_speed']
```


```python
# put both numerical and categorical features together
from scipy import sparse
x = sparse.hstack((dataset3_ohe, dataset3[num_cols]), format='csr')
```


```python
# tip percentage
y = dataset3['tip_percent'] * 100
```

split into training and testing dataset with 80% / 20% rule


```python
np.random.seed(20170429)
train_idx = np.random.choice(x.shape[0], int(x.shape[0] * .8), replace=False)
test_idx = np.ones(x.shape[0], np.bool)
test_idx[train_idx] = False
```


```python
train_x = x[train_idx]
train_y = y.as_matrix()[train_idx]

test_x = x[test_idx]
test_y = y.as_matrix()[test_idx]
```

save processed data for modeling and future reference


```python
with open('./data/train_x.p', 'wb') as handle:
    pickle.dump(train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/train_y.p', 'wb') as handle:
    pickle.dump(train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('./data/test_x.p', 'wb') as handle:
    pickle.dump(test_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('./data/test_y.p', 'wb') as handle:
    pickle.dump(test_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

### Now, our data is ready to be used for modeling :-)
