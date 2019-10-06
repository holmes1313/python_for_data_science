
import pandas as pd

# 1. Combining and Merging Data Sets
'''
Data contained in pandas objects can be combines together in a number of 
built-in ways:

pd.merge connects rows in DataFrame based on one or more keys. ('join')

pd.concat glues or stacks together objects along an axis.

combine_first instance method enables splicing together overlapping data to 
fill in missing values in one object with values from another.'''


## 1.1 Databse-style DataFrame Merges
'''
Merge or join operations combine data sets by linking rows using one or more keys.'''
df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'],
                    'data1': range(7)})
df1

df2 = pd.DataFrame({'key': ['a','b','d'],
                   'data2':range(3)})
df2
    
'''
This is an example of a many-to-one merge situation; the data in df1 has multiple
rows labeled a and b, whereas df2 has only one row for each value in the key
column.'''
pd.merge(df1, df2, on='key')

'''
If you don't specify which column to join on, merge uses the overlapping column
names as the keys. It's a good practice to specify explicitely, though:'''

'''
If the column names are different in each object, you can specify them separately:'''
df3 = pd.DataFrame({'lkey':['b','b','a','c','a','a','b'],
                    'data1':range(7)})
df3

df4 = pd.DataFrame({'rkey': ['a','b','d'],
                   'data2':range(3)})
df4

pd.merge(df3, df4, left_on='lkey', right_on='rkey') 

'''
By default merge does an 'inner' join; the keys in the result are the intersection.
Other possible options are 'left', 'right', and 'outer'.
The outer join takes the union of the keys, combining the effect fo applying
both left and right joins:'''
pd.merge(df1, df2, on='key', how='outer')

'''
Many-to-many merges have well-defined though not necessarily intuitive behavior.
Many-to-many joins form the Cartesian product of the rows.
The join method only affects the distinct key values appearing in the result:'''
df1 = pd.DataFrame({'key':['b','b','a','c','a','b'],
                    'data1':range(6)})
df1
    
df2 = pd.DataFrame({'key':['a','b','a','b','d'],
                    'data2':range(5)})
df2

pd.merge(df1, df2, on='key')

pd.merge(df1, df2, on='key', how='left')


'''
To merge with multiple keys, think of the multiple keys as forming an array
of tuples to be used as a single join key.'''
left = pd.DataFrame({'key1':['foo','foo','bar'],
                     'key2':['one','two','one'],
                     'lval':[1,2,3]})
left

right = pd.DataFrame({'key1':['foo','foo','bar','bar'],
                     'key2':['one','one','one','two'],
                     'rval':[4,5,6,7]})
right

pd.merge(left, right, on=['key1','key2'])
pd.merge(left, right, on=['key1','key2'], how='outer')


'''
A last issue to consider in merge operation is the treatment of overlapping 
column names. While you can address the overlap manually (see the later session
on renaming axis labels), merge has a suffixes option for specifying
strings to append to overlapping names in the left and right DataFrame objects:'''
pd.merge(left, right, on='key1')

pd.merge(left, right, on='key1', suffixes=(['_left', '_right']))
pd.merge(left, right, on='key1', suffixes=('_left','_right'))


## 1.2 Merging on Index
'''
In some cases, the merge key or keys in a DataFrame will be found in its index.
In this case, you can pass left_index=True or right_index=True (or both) to 
indicate that the index should be used as the merge key:'''
left1 = pd.DataFrame({'key':['a','b','a','a','b','c'],
                      'value':range(6)})
left1

right1 = pd.DataFrame({'group_val':[3.5, 7]}, index=['a','b'])
right1

pd.merge(left1, right1, left_on='key', right_index=True)

'''
Since the default merge method is to intersect the join keys, you can instead
form the union of them with an outer join:'''
pd.merge(left1, right1, left_on='key', right_index=True, how='outer')


'''
With hierarchically-indexed data, you have to indicate multiple columns on merge
on as a list (pay attention to the handling of duplicate index values):'''
import numpy as np
lefth = pd.DataFrame({'key1':['Ohio','Ohio','Ohio','Nevada','Nevada'],
                      'key2':[2000, 2001, 2002, 2001, 2002],
                      'data':np.arange(5.0)})
lefth

righth = pd.DataFrame(np.arange(12).reshape(6,2),
                      index=[['Nevada','Nevada','Ohio','Ohio','Ohio','Ohio'],
                             [2001,2000,2000,2000,2001,2002]],
                      columns=['event1','event2'])
righth

pd.merge(lefth, righth, left_on=['key1','key2'], right_index=True)
pd.merge(lefth, righth, left_on=['key1','key2'], right_index=True, how='outer')


'''
Using the indexes of both sides of the merge is also not an issue.'''
left2 = pd.DataFrame([[1,2],[3,4],[5,6]], index=['a','c','e'],
                     columns=['Ohio','Nevada'])
left2

right2 = pd.DataFrame([[7,8],[9,10],[11,12],[13,14]], 
                      index=['b','c','d','e'], columns=['Missouri','Alabama'])
right2

pd.merge(left2, right2, left_index=True, right_index=True)
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)


'''
DataFrame has a more convenient join instance for merging by index.
It can also be used to combine together many DataFrame objects having the
same or similar indexes but non-overlapping columns.'''
left2
right2

left2.join(right2, how='inner')
left2.join(right2, how='outer')
left2.join(right2, how='left')


'''
In part for legacy reasons, DataFrame's join method performs a left join on 
the join keys. It also supports joining the index of the passed DataFrame on 
one of the columns of the calling DataFrame.'''
left1 
right1
left1.join(right1, on='key')


'''
For simple index-on-index merges, you can pass a list of DataFrame to join'''
another = pd.DataFrame([[7,8],[9,10],[11,12],[16,17]],
                       index=['a','c','e','f'], columns=['New York','Oregon'])
another
left2
right2

left2.join([right2, another])
left2.join([right2, another], how='outer')


## 1.3 Concatenating Along an Axis
'''
Numpy has a concatenate function with raw Numpy arrays:'''
import numpy as np

arr = np.arange(12).reshape(3,4)
arr

np.concatenate((arr,arr))
np.concatenate((arr,arr), axis=1)


# the concat function in pandas
'''
Suppose we have three Series with no index overlab:'''
s1 = pd.Series([0,1], index=['a','b'])
s2 = pd.Series([2,3,4], index=['c','d','e'])
s3 = pd.Series([5,6], index=['f','g'])

'''
Calling concat with these object in a list glues together the values and indexes:'''
pd.concat((s1,s2,s3))
pd.concat([s1,s2,s3])

'''
By default concat works along axis=0, producing another Series. If you pass 
axis=1, the result will instead be a DataFrame (axis=1 is the columns***):'''
pd.concat([s1,s2,s3], axis=1)

'''
In this case, there is no overlap on the other axis, which as you can see is 
the sorted union (the 'outer' join) of the indexes. You can instead intersect 
them by passing join='inner':'''
s1
s4 = pd.concat([s1*5, s3])
s4

pd.concat((s1,s4))
pd.concat([s1,s4], axis=1)

pd.concat([s1, s4], axis=1)  # outer by default
pd.concat([s1, s4], axis=1, join='inner')

'''
You can specify the axes to be used with join_axes:'''
pd.concat([s1,s4], axis=1, join_axes=[['a','c','b','e']])


'''
One issue is that the concatenated pieces are not identifiable in the result.
Suppose instead you wanted to create a hierarchical index on the concatenation 
axis. To do this, use the keys argument:'''
result = pd.concat([s1,s2,s3], keys=['one','two','three'])
result

result.unstack()

'''
In the case of combining Series along axis=1, the keys become the DataFrame 
column headers:'''
pd.concat([s1,s2,s3], axis=1, keys=['one','two','three'])

'''
The same logic extends to DataFrame objects:'''
df1 = pd.DataFrame(np.arange(6).reshape(3,2), index=['a','b','c'],
                   columns=['one', 'two'])
df1

df2 = pd.DataFrame(5 + np.arange(4).reshape(2,2), index=['a','c'],
                 columns=['three','four'])
df2

pd.concat([df1,df2])
pd.concat([df1,df2], axis=1, keys=['level1','level2'])

'''
If you pass a dict of objects instead of a list, the dict's keys will be used
for the keys option:'''
pd.concat({'level1':df1, 'level2':df2}, axis=1)

'''
There are a couple of additional arguments governing how the hierarchical index
is created:'''
pd.concat([df1, df2], axis=1, keys=['level1','level2'],
          names=['upper','lower'])


'''
A last consideration concerns DataFrames in which the row index is not meaningful
in the context of the analysis. In this case, you can pass ignore_index=True:'''
df1 = pd.DataFrame(np.random.randn(3,4), columns=['a','b','c','d'])
df1

df2 = pd.DataFrame(np.random.randn(2,3), columns=['b','d','a'])
df2

pd.concat([df1,df2], ignore_index=True)


## 1.4 Combining Data with Overlap
'''
You may have two datasets whose indexes overlap in full or part.
As a motivating example, consider Numpy's where function, which expressed a 
vectorized if-else:'''
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
              index=['f','e','d','c','b','a'])
a

b = pd.Series([0, 1.0, 2.0, 3.0, 4.0, 5.0],
              index=['f','e','d','c','b','a'])
b[-1] = np.nan
b

np.where(a.isnull(),b,a)
np.where(pd.isnull(a), b, a)


'''
Series has a combine_first method, which performs the equivalent of this operation
plus data alignment:'''
b
a
a.combine_first(b)

b[:-2].combine_first(a[2:])


'''
With DataFrame, combine_first naturally does the same thing column by column,
so you can think it as 'patching' missing data in the callng object with 
data from the object you pass:'''
df1 = pd.DataFrame({'a':[1., np.nan, 5., np.nan],
                    'b':[np.nan, 2., np.nan, 6.],
                    'c':range(2,18,4)})
df1
    
    
df2 = pd.DataFrame({'a':[5., 4., np.nan, 3., 7.],
                    'b':[np.nan, 3., 4., 6., 8.]})
df2    
    
df1.combine_first(df2)



## 2. Reshaping and Pivoting
# 2.1 Reshaping with Hierarchical Indexing
'''
Hierarchical indexing provides a consistent way to rearrange data in a DataFrame.
-stack: this 'rotates' or pivots from the columns in the data to the rows

-unstack: this pivots from the rows into the columns'''
data = pd.DataFrame(np.arange(6).reshape(2,3),
                    index=pd.Index(['Ohio','Colorado'], name='state'),
                    columns=pd.Index(['one','two','three'], name='number'))
data

'''
Using the stack method on this data pivots the columns into the rows, producting
a Series:'''
result = data.stack()
result
result.unstack()

data
data.unstack()

'''
From a hierarchically-indexed Series, you can rearrange the data back into a
DataFrame with unstack:'''
result.unstack()

'''
By default the innermost level is unstacked (same with stack). You can unstack
a different level by passing a level number or name:'''
result
result.unstack()
result.unstack(0)
result.unstack('state')

'''
Unstacking might introduce missing data if all of the values in the level aren't
found in each of the subgroups:'''
s1 = pd.Series([0,1,2,3], index=['a','b','c','d'])
s2 = pd.Series([4,5,6], index=['c','d','e'])
data2 = pd.concat([s1,s2], keys=['one','two'])
data2

data2.unstack()
data2.unstack(0)

'''
Stacking filters out missing data by default, so the operation is easily 
invertible:'''
data2
data2.unstack().stack()

'''
When unstacking in a DataFrame, the level unstacked becomes the lowest in the 
reuslt:'''
result
df = pd.DataFrame({'left':result, 'right':result+5},
                  columns=pd.Index(['left','right'], name='side'))
df

df.unstack('state')

df.unstack('state').stack('state')
df.unstack('state').stack('side')

df.unstack()
df.stack()


# 2.2 Pivoing 'long' to 'wide' format  help!
'''
A common way to store multiple time series in databases and csv is in so-called
long or stacked format:'''
data = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python' \
                   '\pyhton_for_data_science\macrodata.txt')
data.head()
data.shape

periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
type(periods)
periods.to_timestamp('D','end')

type(data.to_records())
data = pd.DataFrame(data.to_records(),
                    columns = pd.Index(['realgdp','infl','unemp'],name='item'),
                    index = periods.to_timestamp('D','end'))

data.head()

data.stack()
data.stack().reset_index()
ldata = data.stack().reset_index().rename(columns={0:'value'})
ldata


'''
You might prefer to have a DataFrame containing one column per distinct item
value indexed by timestamps in the date column. DataFrame's  pivot method performs
exactly this transformation:'''
ldata.head()

pivoted = ldata.pivot('date','item','value')
pivoted.head()

'''
The first two values passed are the columns to be used as the row and columns index,
and finally an optional value column to fill the DataFrame.
Suppose you had two value columns that you want to reshape simultaneoutly:'''
len(ldata)
ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]

'''
By omitting the last argument, you obtain a DataFrame with hierarchical columns:'''
pivoted = ldata.pivot('date','item')
pivoted[:5]

pivoted['value'][:5]

'''
Note that pivot is just a shortcut for creating a hierarchical index using 
set_index and reshaping with unstack:'''
ldata.head()

unstacked = ldata.set_index(['date','item']).unstack('item')
unstacked.head()



## 3. Data Transformation

# 3.1 Removing Duplicates
'one'*3
['one']*3    

'one'+'two'    
['one']+['two']  

data = pd.DataFrame({'k1':['one']*3 + ['two']*4,
                     'k2':[1,1,2,3,3,4,4]})
data
    
'''
The DataFrame method duplicated returns a boolean Series indicating whether each
row is a duplicate or not:'''
data.duplicated()

'''
Relatedly, drop_duplicates returns a DataFrame where the duplicated array is 
False:'''
data.drop_duplicates()
data.drop_duplicates().reindex(range(len(data)))

'''
Both of these methods by default consider all of the columns; alternatively you 
can specify any subset of them to detect duplicates.
Suppose we had an additional column of values and wantted to filter duplicates
only based on the 'k1' column:'''    
data['v1'] = np.arange(len(data))
data    

data.drop_duplicates(['k1'])    
    
'''
duplicated and drop_duplicates by default keep the first observed value combination.
Passing take_last=True will return the last one:'''
data
data.drop_duplicates(['k1','k2'], keep='last')
    
  
# 3.2 Transforming Data Using a Function or Mapping
data = pd.DataFrame({'food':['bacon','pulled pork','bacon','Pastrami',
                             'corned beef', 'Bacon','pastrami','honey ham',
                             'nova lox'],
                    'ounces':[4,3,12,6,7.5,8,3,5,6]})
  
'''
Suppose you wanted to add a column indicating the type of animal that each food
came from. Let's write down a mapping of each distinct meat type to the kind of 
animal:'''
meat_to_animal = {
        'bacon':'pig',
        'pulled pork':'pig',
        'pastrami':'cow',
        'corned beef':'cow',
        'honey ham':'pig',
        'nova lox':'salmon'
        }
    

'''
The map method on a Series accepts a function or dict-like object containing a 
mapping.
Here we have a small problem in that some of the meats above are capitalized
and other are not. Thus, we also need to convert each value to lower case:'''
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
    
meat_to_animal2 = {
        'pig':'bacon',
        'pig':'pulled pork',
        'cow':'pastrami',
        'cow':'corned beef',
        'pig':'honey ham',
        'salmon':'nova lox'
        }    
data['food'].map(str.lower).map(meat_to_animal2)
    
'''
We could also have passed a function that does all the work:'''
data['food'].map(lambda x: meat_to_animal[x.lower()])
    
'''
Using map is a convenient way to perform element-wise transformations and other
data cleaning-related operations.'''

    
# 3.3 Replacing Values
'''
Filling in missing data with the fillna method can be thought of as a special
case of more general value replacement.
While map can be used to modify a subset of values in an object, replace provides
a simpler and more flexible way to do so.'''
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data

'''
The -999 values might be sentinel values for missing data. To replace these
with NA values that pandas understands, we can use replace, producing a new
Series:'''
data.replace(-999, np.nan)   # the original data keeps the same
data

data[data==-999] = data[data==-999].map(lambda x: np.nan)
data

data[data==-999] = np.nan
data

'''
If you want to replace multiple values at once, you instead pass a list then the
substitute value:'''
data.replace([-999, -1000], np.nan)

'''
To use a different replacement for each value, pass a list of substitutes:'''
data.replace([-999, -1000], [np.nan, 0])

'''
The argument passed can also be a dict:'''
data.replace({-999:np.nan, -1000:0})



# 3.4 Renaming Axis Indexes
'''
Like values in a Series, axis labels can be similarly transformed by a function
or mapping of some form to produce new, differently labeled objects.
The axes can also be modified in place without creating a new data structure.'''
data = pd.DataFrame(np.arange(12).reshape(3,4),
                    index=['Ohio','Colorado','New York'],
                    columns=['one','two','three','four'])
data

'''
Like a Series, the axis indexes have a map method:'''
data.index.map(str.upper)
data.columns.map(str.upper)

'''
You can assign to index, modifying the DataFrame in place:'''
data.index = data.index.map(str.upper)
data

'''
If you want to create a transformed version of a data set without modifying
the orignal, a useful method is rename:'''
data.rename(index={'OHIO':'Ohio'})
data.rename(index=str.title, columns=str.upper)


'''
Notably, rename can be used in conjunction with a dict-like object providing
new values for a subset of the axis labels:'''
data.rename(index={'OHIO':'INDIANA'},
            columns={'three': 'peekaboo'})
data
'''
rename saves having to copy the DataFrame manually and assign to its index and
columns attributes. 
Should you wish to modify a data set in place, pass inplace=True:'''
# Always return a reference to a DataFrame
data.rename(index={'OHIO':'INDIANA'},
            columns={'four':'heihei'}, inplace=True)
data



# 3.5 Discretization and Binning
'''
Continuous data is often discretized or otherwise separated into 'bins' for analysis.
Suppose you want to group some ages into discrete age buckets:'''
ages = [20,22,25,27,21,23,37,31,61,45,41,32]

'''
Let's divide these into bins of 18 to 25, 26 to 35, 36 to 60, and finally 61 and
older. To do so, you have to use cut, a function a pandas:'''
bins = [18, 25, 35, 60, 100]

cats = pd.cut(ages, bins)
cats   # categorical object

'''
The object pandas returns is a special Categorical object. You can treate it like
an array of strings indicating the bin name;
initially it contains a levels array indicating the distinct category names along 
with a labeling for the ages data in the labels attribute:'''
cats.labels

cats.value_counts()
pd.value_counts(cats)

'''
Which side is closed or open can be changed by passing right=False:'''
pd.cut(ages, bins, right=False)

'''
You can also pass your own bin names by passing a list or array to the labels
option:'''
group_names = ['youth','youngadult','middleage','senior']

cats2 = pd.cut(ages, bins, labels=group_names)
pd.value_counts(cats2)

'''
If you pass cut a integer number of bins instead of explicit bin ages, it will
compute equal-length bins based on the minimum and maximum values in the data.

Consider the case of some uniformly distributed data chopped into fourths:'''
data=np.random.rand(20)
data
data.min()
np.sort(data)
data

cats3 = pd.cut(data, 4, precision=2) #equal-length bins 
pd.value_counts(cats3)


'''
A closely related funciton, qcut, bins the data based on sample quantiles, and
you will obtain roughly equal-size bins:'''
data = np.random.randn(100)  # normal distibuted

cats4 = pd.qcut(data, 4) # cut into quartiles
pd.value_counts(cats4)

'''
Similar to cut you can pass your own quantitles (numbers between 0 and 1, 
inclusive):'''
data = np.random.randn(100)  # normal distibuted                                                 

cats5 = pd.qcut(data, [0,0.1,0.5,0.9,1.])
pd.value_counts(cats5, ascending=True)

'''
Discretization functions are especially useful for quantile and group analysis.'''



# 3.6 Detecting and Filering Outliers
'''
Filtering or transforming outliers is largely a matter of applying array 
operations. Consider a DataFrame with some normally distributed data:'''
np.random.seed(12345)

data = pd.DataFrame(np.random.randn(1000,4))

data.head()
data.describe()

'''
Suppose you wannted to find values in the third columns exceeding three in 
magnitude:'''
col = data[3]
np.abs(col) > 3
col[np.abs(col)>3]

'''
To select all rows having a value exceeding 3 or -3, you can use the any method
on a boolean DataFrame:'''
data[(np.abs(data)>3).any(1)]


'''
Values can just as easily be set based on these criteria. Here is code to cap
values outside the interval -3 to 3:'''
data[np.abs(data)>3] = np.sign(data) * 3
data
data.describe()

'''
the ufunc np.sign returns an array of 1 and -1 depending on the sign of the values.'''



# 3.7 Permutation and random Sampling
'''
Permutation (randomly reordering) a Series or the rows in a DataFrame is easy 
to do using the np.random.permutation function. Calling permutation with the 
length of the axis you want to permute produces an array of integers indicating
the new ordering:'''
df = pd.DataFrame(np.arange(5*4).reshape(5,4))
df

np.random.permutation(5)
sampler = np.random.permutation(5)
sampler

df.iloc[sampler]

df[np.random.permutation(4)]

df.take(sampler).reset_index()
df


'''
To select a random subset without replacement, one way is to slice off the first k 
elements of the array returned by permutation, where k is the desired subset size.
There are much more efficient sampling-without-replacement algorithms, but this 
is an easy strategy that uses readily available tools:'''

df.iloc[np.random.permutation(len(df))][:3]
df.iloc[np.random.permutation(len(df))[:3]]

'''
To generate a sample with replacement, the faster way is to use np.random.randint
to draw random integers:'''
bag = np.array([5,7,-1,6,4])

sampler = np.random.randint(0,len(bag),size=10)
sampler

draws = bag[sampler]
draws


df = pd.DataFrame(np.arange(5*4).reshape(5,4))
df
sampler = np.random.randint(0, len(df), size=10)
sampler
df.iloc[sampler]  # with replacement


# 3.8 Computing Indicator/Dummy Variables
'''
Converting a categorical variable into a 'dummy' or 'indicator' matrix.
If a column in a DataFrame has k distinct values, you would derive a matrix
or DataFrame containing k columns containing all 1's and 0's.
Pandas has a get_dummies function for doing this.'''
df = pd.DataFrame({'key':['b','b','a','c','a','c'],
                   'data1':range(6)})
df

pd.get_dummies(df['key'])


'''
In some cases, you may want to add a prefix to the columns in the indicator
DataFrame, which can then be merged with the other data. 
get_dummies has a prefix argument for doing just this:'''
dummies = pd.get_dummies(df['key'], prefix='key')
dummies

df_with_dummy = df[['data1']].join(dummies)
df_with_dummy

######################################################  unsolved p203
'''
If a row in a DataFrame belongs to multiple categories,'''
mnames = ['movie_id','title','genres']

movies = pd.read_table(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science'\
                       '\movies.dat.txt', sep='::', header=None, names=mnames)
movies.head()

'''
Adding indicator variables for each genre requires a little bit of wrangling.
First, we extraxct the list of unique genres in the dataset 
(using a nice set.union trick):'''

set(movies['genres'][1].split('|'))

genre_iter = [set(x.split('|')) for x in movies['genres']]

genres = set.union(*genre_iter)

'''
One way to construct the indicator DataFrame is to start with a DataFrame of 
all zeros:'''
dummies = pd.DataFrame(np.zeros((len(movies), len(genres))), columns=genres)
dummies

'''
Iterate through each movie and set entries in each row of dummies to 1'''

for i, gen in enumerate(movies['genres']):
    if i < 6:
        print(i, gen)

for i, gen in enumerate(movies['genres']):
    dummies.loc[i, gen.split('|')] = 1

dummies.head()

'''
Combine this with movies'''
movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.head()

#######################################################


'''
A usefull recipe for statistical application is to combine get_dummies with
a discretization function like cut:'''
values = np.random.rand(10)
values

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

pd.cut(values, bins)

pd.get_dummies(pd.cut(values, bins))


# 4. String Manipulation
## 4.1 String Object Methods
'''
a comma-separated string can be broken into pieces with split:'''
val = 'a,b, guido'
val.split(',')

'''
split is often combined with strip to trim whitespace (including newlines):'''
pieces = [x.strip() for x in val.split(',')]

'''
val.split(',').strip()
'list' object has no attribute 'strip''''

'''
These substrings could be concatenated together with a two-colon delimiter using
addition:'''
first, second, third = pieces

first + '::' + second + '::' + third

'''
A faster and more Pythonic way is to pass a list or tuple to the join method
on the string '::':'''
pieces
'::'.join(pieces)

'''
Other methods are concerned with locating substrings.
Using Python's in keyword is the best way to detect a substring, though
index and find can also be used:'''
val
'guido' in val
'o' in val

val.index(',')

val.find(':')

'''
Note the different between find and index is that index raises an exception
if the string isn't found (versus returning -1)'''
val.index(':')
# ValueError: substring not found

'''
count returns the numbers of occurences of a particular substring:'''
val
val.count('o')
val.count(',')

'''
replace will substitute occurances of one pattern for another. 
This is commonly used to delete patterns, too, by passing an empty string:'''
val.replace(',', '::')

val.replace(',','').replace(' ','')



# Table 7-3. Python built-in string methods
val1 = 'Jeremy'
val2 = ' is wonderful!'

# endswith, startswith
val2.startswith(' is')

# lower, upper
val2.upper()


## 4.2 Regular expression
'''
Regular expressions provide a flexible way to search or match string patterns
in text. A single expression, commomly called a regex, is a string formed 
according to the regular expression language. Python's build-in re module is 
responsible for applying regular expressions to strings.

The re module functions fall into three categories: pattern matching, substituion,
and splitting.
A regex describes a pattern to locate in the text, which can then be used for 
many purposes.

Suppose I wantted to split a string with a variable number of whitespace characters
(tabs, spaces, and newlines). The regex describeing one or more whitespaces 
characters is \s+:'''
import re

text = 'food   bar\t baz  \tqux'
print(text)

re.split('\s+', text)

text.split(' ')
text.split()
text.split('\s+')

'''
When you call re.split('\s+',text), the regular expression is first compiled,
then its split method is called on the passed text. You can compile the regex
yourself with re.complie, forming a resusable regex object:'''
regex = re.compile('\s+')

regex.split(text)

'''
If, instead, you wanted to get a list of all patterns matching the regex, 
you can use the findall method:'''
regex.findall(text)

'''
While findall returns all matches in a string, search returns only the first match.
More rigidly, match only matches at the beginning of the string. As a less trivial
example, let's consider a block of text and a regular expression capable of identifying
most email addresses:'''

text = """Dave dave@gmail.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@gmail.com
"""

pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

# re.IGNORECASE makes the regex case-insentitive
regex = re.compile(pattern, flags=re.IGNORECASE)

'''
Using findall on the text produces a list of the e-mail addresses:'''
regex.findall(text)

'''
search returns a special match object for the first email address in the text.
For above regex, the match object can only tell us the start and end position
of the pattern in the string:'''
m = regex.search(text)
m

text[m.start():m.end()]


########################################################### unsovled p207


## 4.3 Vectorized string functions in pandas
data = {'Dave':'dave@google.com', 'Steve':'steve@gmail.com',
        'Rob':'rob@gmail.com','Wes':np.nan}
data = pd.Series(data)
data

data.isnull()
data.notnull()

'''
Series has concise methods for string operations that skip NA values.
These are accessed through Series's str attribute; for example, we could 
check whether each email address has 'gmail' in it with str.contains:'''
data.str

data.str.contains('gmail')

'''
Regular expressions can be used, too, along with any re options like IGNORECASE:'''
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

data.str.findall(pattern, flags=re.IGNORECASE)

'''
There are a couple of ways to do vectorized element retrieval.'''
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches

'''
You can similarly slice strings using this syntax'''
data.str[:5]




# 5. Example: USFA Food Database  unsovled
import json

db = json.load(open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science\foods-2011-10-03.json.txt'))

len(db)
type(db)

db[0].keys()
type(db[0])

db[0]['nutrients'][0]
type(db[0]['nutrients'])

nutrients = pd.DataFrame(db[0]['nutrients'])
nutrients.head()
nutrients.shape

db[0].keys()
info_keys = ['description', 'group','id','manufacturer']
info = pd.DataFrame(db, columns=info_keys)
info.head()
info.shape
info.info()

info['group'].value_counts()

nutrients = []

for rec in db:
    funts = pd.DataFrame(rec['nutrients'])
    funts['id'] = rec['id']
    nutrients.append(funts)

nutrients[0]
type(nutrients)
type(nutrients[0])

nutrients = pd.concat(nutrients, ignore_index=True)
type(nutrients)

nutrients.duplicated().sum()
nutrients = nutrients.drop_duplicates()

col_mapping = {'description':'food',
               'group':'fgroup'}
info = info.rename(columns=col_mapping)
info.columns
info.info()


col_mapping = {'description':'nutrient',
               'group':'nutgroup'}
nutrients = nutrients.rename(columns = col_mapping)
nutrients.info()

ndata = pd.merge(nutrients, info, on='id', how='left')
ndata.info()
ndata.iloc[30000]

'''median values by food group and nutrient type'''
result = ndata.groupby(['nutrient','fgroup'])['value'].median()
type(result)
result

result['Zinc, Zn']['Baby Foods']
result['Zinc, Zn']
result['Zinc, Zn'].sort_values().plot(kind='barh')

'''
Find which food is most dense in each nutrient:'''
by_nutrient = ndata.groupby(['nutrient','fgroup'])

get_maximum = lambda x: x.xs(x.value.idxmax())

by_nutrient.apply(get_maximum)[['value','food']]



'''
Each entry in db is a dict containing all the data for a single food. The 'nutrients'
field is a list of dicts, one for each nutrient:'''
db[0].keys()

len(db[0]['nutrients'])
db[0]['nutrients'][0]

nutrients = pd.DataFrame(db[0]['nutrients'])
nutrients.head(7)

'''
When converting a list of dicts to a DataFrame, we can specify a list of fields
to extract. We'll take the food names, group, id, and manufacturer:'''
info_key = ['description', 'group','id','manufacturer']

info = pd.DataFrame(db, columns=info_key)
info.head()
info.info()

'''
You can see the distribution of food groups with value_counts:'''
pd.value_counts(info.group)[:10]

'''
Now, to do some analysis on all of the nutrient data, it's easiest to assemble 
the nutrients for each food into a single large table.
To do so, we need to take several steps.
First, I'll convert each list of food nutrients to a DataFrame, add a columns 
for the food id, and append the DataFrame to a list. Then, these can be concatenated
together with concat:'''
nutrients = []

for rec in db:
    funts = pd.DataFrame(rec['nutrients'])
    funts['id'] = rec['id']
    nutrients.append(funts)

len(nutrients)

nutrients = pd.concat(nutrients, ignore_index=True)
nutrients.shape

'''
I noticed that, for whatever reason, there are duplicates in this DataFrame, 
so it makes things easier to drop them:'''
nutrients.duplicated().sum()

nutrients = nutrients.drop_duplicates()
nutrients.shape

'''
Since 'group' and 'description' is in both DataFrame objects, we can rename them
to make it clear what is what:'''
col_mapping = {'description':'food',
               'group':'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
info.info()


col_mapping = {'description':'nutrient',
               'group':'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
nutrients.info()

'''
With all of this done, we're ready to merge info with nutrients:'''
ndata = pd.merge(nutrients, info, on='id', how='outer')

ndata.info()

ndata.iloc[30000]

'''
We could a plot of median values by food group and nutrient type:'''
result = ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)

result['Zinc, Zn'].sort_values().plot(kind='barh')

'''
With a little cleverness, you can find which food is most dense in each nutrient:'''
by_nutrient = ndata.groupby(['nutgroup','nutrient'])

get_maximum = lambda x: x.xs(x.value.idxmax())
get_minimum = lambda x: x.xs(x.value.idxmin())

max_foods = by_nutrient.apply(get_maximum)[['value','food']]

# make the food a  little smaller
max_foods.food = max_foods.food.str[:50]

'''
The resulting DataFrame is a bit too large to display in the book; here is just
the 'Amino Acids' nutrient group:'''
max_foods.loc['Amino Acids']['food']

max_foods













