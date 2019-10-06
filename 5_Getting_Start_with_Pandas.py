

"""
Pandas is built on the top of Numpy and makes it easy to use in NumPy-centric
applications.
"""
import numpy as np
import pandas as pd

# 5.1 Introduction to pandas data structures

## 5.1.1 Series
'''
A Series is a one-dimensional array-like object containing an array of data
(of any NumPy data type) and an associated array of data labels, called its
index.'''
obj = pd.Series([4,7,-5,3])

'''
You can get the array representation and index object of the Series via its
values and index attributes, respectively:'''
obj.values
obj.index

'''
To create a Series with an index identifying each data point:'''
obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])
obj2.index


'''
To use values in the index when selecting single values or a set of values:'''
obj2
obj2['a']
obj2.a
obj2[2]

obj2['d'] = 6  ## change the value of index 'd' from 4 to 6
obj2

obj2[['c','a','d']]

'''
Numpy array operations, such as filtering with a boolean array, scalar multiplication,
or applying math function, will preserve the index-value link:'''
obj2 > 2
obj2[obj2>2]

obj2 * 2

np.exp(obj2)

'''
Another way to think about a Series is as a fixed-length, ordered dict, 
as it is a mapping of index values to data values.'''
'b' in obj2.index

'e' in obj2.index

'''
Create a Series by passing a Pyhton dict
When only passing a dict, the index in the resulting Series will have 
the dict's keys in sorted order.'''

sdata = {'Ohio':35000,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3 = pd.Series(sdata)
obj3
'''
When only passing a diect, the index in the resulting Series will have the dict's
keys in sorted order.'''

states = ['California','Ohio','Texas','Oregon']
obj4 = pd.Series(sdata, index=states)
obj4
'''
In obj4, 3 values found in sdata were placed in the given sequence, but NaN(
not a number) for 'California' is found.'''

        
'''
Use isnull and notnull in Pandas to detect missing data:
Series also has theses as instance methods
'''
obj4 
obj4.isnull()
obj4.notnull()

pd.isnull(obj4)
pd.notnull(obj4)

'''
A critical Series feature for many applications is that it automatically aligns 
differently indexed data in arithmetic operations:'''
obj3
obj4
obj3 + obj4
obj3.add(obj4, fill_value=0).fillna(0)

'''
Both the Seires object itself and its index have a name attribute, which 
integrates with other key areas of pandas functionality:'''
obj4
obj4.name='population'
obj4.index.name='states'
obj4

'''
A Series's index can be altered in place by assignment:'''
obj
obj.index = ['a','b','c','d', 'e']
obj


## 5.1.2 DataFrame
'''
A DataFrame represents a tabular, spreadsheet-like data structure containing
an ordered collection of columns, each of which can be a different value type.

one of the most common way to construct a DataFrame is from 
a dict of equal-length lists ro Numpy arrays.'''
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
        'year':[2000,2001,2002,2001,2002],
        'pop':[1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(data)
frame

'''
the columns are placed in sorted order, but you can specify a sequence of columns'''
pd.DataFrame(data, columns=['year','state','pop'])


'''
As with Series, if you pass a column that isn't contained in data, 
it will appear with NaN values in the result:'''
frame2 = pd.DataFrame(data, columns=['year','state','pop','debt'],
                            index=['one','two','three','four','five'])
frame2

frame2.columns
frame2.index

'''
Retrieve a column from DataFrame:'''
frame2.state
frame2['state']
frame2.iloc[:,1]

'''
Rows van be retrieved by position or name by a couple of methods, such as 
the ix indexing field:'''
frame2
frame2.loc['three']
frame2.iloc[2]

'''
Columns can be modified by assignment. For example, the empty 'debt' column
could be assigned a scalar value or an array values:'''
frame2

frame2.debt = 16.5
frame2

frame2['debt'] = np.arange(5)
frame2

'''
When assigning lists or arrays to a column, the value's length must match 
the length of the DataFrame.
But if you assign a Series, it will be instead conformed exactly to the 
DataFrame's index, inserting missing values in any holes:'''
frame2

val = pd.Series([-1.2, -1.5, -1.7], index=['two','four','five'])
val
frame2['debt'] = val
frame2


'''
Assigning a column that does't exist will create a new column.
The del keyword will delete columns as with a dict:'''
frame2
frame2['eastern'] = frame2['state']=='Ohio'
frame2

del frame2['eastern']
frame2
frame2.columns


'''
Another common form of data is a nested dict of dicts format.
If passed to DataFrame, it will interpret the outer dict keys as the columns
and the inner keys as the row indices:'''
pop = {'Nevada':{2001:2.4, 2002:2.9},
       'Ohio':{2000:1.5, 2001:1.7, 2002:3.6}}
frame3 = pd.DataFrame(pop)
frame3
pd.Series(pop)

'''
you can always transpose the result:'''
frame3.T
frame3.unstack()
frame3.stack()
frame3

'''
Dicts of Series are treated much in the same way:'''
frame3
frame3['Ohio'][:-1]
frame3.iloc[:-1, -1]
frame3.loc[:2001,'Ohio']

frame3['Nevada'][:2]
frame3.loc[:2001, 'Nevada']
frame3.iloc[:2,0]

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pdata
pd.DataFrame(pdata)


'''
the name attributes set of a DataFrame's index and columns:'''
frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3

frame3.name = 'test'


'''
The values attribute returns the data contained in the DataFrame as
a 2D ndarray:'''
frame3.values

'''
If the DataFrame's columns are different dtypes, the dtype of the values 
array will be chosen to accomodate all of the columns:'''
frame2.values



## 5.1.3 Index Objects
'''
Pandas's Index objects are responsible for holding the axis labels
and other metadata (like the axis name or names).
Any array or other sequence of labels used when constructing a Series or DataFrame
is initially converted to an Index:'''
obj = pd.Series(np.arange(3), index=['a','b','c'])
obj

index = obj.index
index
index[1:]

'''
Index objexts are immutable and thus can't be modified by the users:'''
index[1] = 'd'
# TypeError: Index does not support mutable operations

'''
Immutability is important so that Index objects can be safely shared 
among data structuer:'''
index = pd.Index(np.arange(3))
index
obj2 = pd.Series([1.5, -2.5, 0], index=index)
obj2

obj2.index is index
# True

'''
In addition to being array-like, an Index also functions as a fixed-size 
set:'''
frame3

'Ohio' in frame3.columns
# True
2003 in frame3.index
# False


# 5.2 Essential Functions
## 5.2.1 Reindexing
'''
reindex is to create a new object with the data conformed to a new
index.'''
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d','b','a','c'])
obj
'''
Calling reindex on thie Series rearranges the data according to the new
index, introducing missing values if any index values were not already 
present:'''
obj2 = obj.reindex(['a','b','c','d','e'])
obj2

obj.reindex(['a','b','c','d','e'], fill_value=0)

'''
The 'method' option allows us to do some interpolation or filling, 
using a method such as 'ffill' which forward fills the values:'''
obj3 = pd.Series(['blue','purple','yellow'],index=[0,2,4])
obj3

obj3.reindex(range(6), method='ffill')
obj3.reindex(range(9), method='bfill')

'''
With DataFrame, reindex can alter either the (row) index, columns, or both.
'''
frame = pd.DataFrame(np.arange(9).reshape((3,3)), index=['a','c','d'],
                     columns=['Ohio','Texas','California'])
frame

frame2 = frame.reindex(index=['a','b','c','d'])
frame2

frame.reindex(columns=['Texas','Utah','California'])
frame

'''
Both can be reindexed in one shot, though interpolation will only apply
row-wise (axis 0):'''
frame.reindex(index=['a','b','c','d'], columns=['Texas','Utah','California'])
## ??? ValueError if there is method='ffill'

'''
reindexing can be done more succinctly by label-indexing with ix:'''
frame.loc[['a','b','c','d'], ['Texas','Utah','California']]


## 5.2.2 Dropping entries from an anxis
'''
The drop method can return a new object with the indicated value or 
values deleted from an axis:'''
obj = pd.Series(np.arange(5.),index=['a','b','c','d','e'])
obj

del obj['c']
obj

new_obj = obj.drop('c')
new_obj
obj

obj.drop(['d','c'])


'''
With DataFrame, index values can be deleted from either axis 
(row:axis=0; column:axis=1):'''
data = pd.DataFrame(np.arange(16).reshape((4,4)),
                 index=['Ohio','Colorado','Utah','New York'],
                 columns=['one','two','three','four'])
data

data.drop(['Colorado'])
data.drop(['one','two'], axis=1)

data.drop(['Colorado', 'Ohio'])
data.drop('two',axis=1)
data.drop(['two','four'],axis=1)


## 5.2.3 Indexing, selection, and filtering
'''
Series indexing (obj[...]) works analogously to numpy array indexing, except
you can use Series's index values instead of only integers.'''
obj = pd.Series(np.arange(4.), index=['a','b','c','d'])
obj

obj['b']
obj[1]
obj[2:4]
obj[['b','a','d']]
obj[[1,3]]
obj[obj<2]

'''
Slicing with labels behaves differently than normal python slicing in that
the endpoint is inclusive:'''
obj
obj['b':'c']

obj['b':'c']= 5
obj

'''
indexing into a DataFrame is for retrieving one or more columns either with a
single value or sequence:'''
data = pd.DataFrame(np.arange(16).reshape((4,4)),
                    index=['Ohio','Colorado','Utah','New York'],
                    columns=['one','two','three','four'])
data

data['two']
data[['two']]
data[['three','one']]

'''
Selecting rows by slicing or a boolean array:'''
data[:2]
data.iloc[:2]

data[data['three']>5]

'''
Another use case is in indexing with a boolean DataFrame, such as one 
produced by a scalar comparison'''
data < 5

data[data<5] = 0
data

'''
For DataFrame label-indexing on the rows, loc enables you to select a subset
of the rows and columns from a DataFrame with NumPy like notation plus
axis lables.'''
data

data.loc['Colorado',['two','three']]
data.loc[['Colorado'],['two','three']]

data.loc[['Colorado','Utah'],['four','one','two']]
data.ix[['Colorado','Utah'],[3,0,1]]

data.iloc[[2]]

data.loc[:'Utah','two']

data.loc[data.three>5, ['one','two','three']]
data.ix[data.three>5,:3]
data[data.three>5][['one','two','three']]



## 5.2.4 Arithmetic and data alignment
'''
One of the most important pandas features is the behavior of arithmetic between
objects with different indexes. When adding together objects, if any index pairs
are not the same, the respective index in the result will be the union of the index
pairs.'''
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a','c','d','e'])
s1
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a','c','e','f','g'])
s2

s1 + s2
'''
The internal data alignment introduces NA values in the indices that don't
overlap.'''


'''
In the case of DataFrame, alignment is performed on both the rows and the 
columns:'''
df1 = pd.DataFrame(np.arange(9.).reshape((3,3)), columns=list('bcd'),
                    index=['Ohio','Texas','Colorado'])
df1
df2 = pd.DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'),
                   index=['Utah','Ohio','Texas','Oregon'])
df2

df1 + df2

# Arithmatic methods with fill values
'''
To fill with a special value, like 0, when an axis label is found in one object
but not the other, you can use the add method on df1 and pass df2 and an 
argument to fill_value:'''
df1 = pd.DataFrame(np.arange(12.).reshape(3,4), columns=list('abcd'))
df1

df2 = pd.DataFrame(np.arange(20.).reshape(4,5), columns=list('abcde'))
df2

df1 + df2

'''
Using the add method on df1, I pass df2 and an argument to fill_value:'''
df1.add(df2, fill_value=0)

'''
Relatedly, when reindexing a Series or DataFrame, you can also specify a different fill
value:'''
df1.reindex(columns=df2.columns, fill_value=0)

'''
Table 5-7. Flexible arithmetic methods:
    add, sub, div, mul'''


# Operations between DataFrame and Series
'''
First, consider the differentce between a 2D array and one of its rows:'''
arr = np.arange(12.).reshape(3,4)
arr

arr[0]

arr - arr[0]

'''
Operations between a DataFrame and a Series are similar:'''
frame = pd.DataFrame(np.arange(12).reshape(4,3), columns=list('bde'),
                     index=['Utah','Ohio','Texas','Oregon'])
frame

series = frame.iloc[0]
series

'''
By default, arithmetic between DataFrame and Series matches the index of the 
Series on the DataFrame's columns, boardcasting down the rows:'''
frame - series

series1 = pd.Series([1,2], index=['b','d'])
series1
frame - series1

'''
If an index value is not found in either the DataFrame's columns or the 
Series's index, the objects will be reindexed to form the union:'''
series2 = pd.Series(range(3), index=['b','e','f'])
series2

frame + series2

'''
If you want to instead broadcast over the columns, matching on the rows,
you have to use one of the arithmetic methods.'''
series3 = frame['d']
series3

frame
frame.sub(series3, axis=0)
frame - series3

'''
The axis number that you pass is the axis to match on. In this case we mean to
match on the DataFrame's row index and broadcasr across.'''

    
## 5.2.5 Function application and mapping
'''
NumPy ufuncs (element-wise array methods) work fine with pandas objects:'''
frame = pd.DataFrame(np.random.randn(4,3), columns=list('bde'),
                     index=['Utah','Ohio','Texas','Oregon'])
frame

np.abs(frame)

'''
Another frequent operation is applying a function on 1D array to each column or 
row. DataFrame's apply method does exactly this:'''
f = lambda x: x.max() - x.min()
frame.apply(f) # apply to columns
frame.apply(f, axis=1) # apply to rows

f = lambda x: pd.Series([x.min(), x.max()], index=['min','max'])
frame.apply(f)
frame.apply(f, axis=1)


'''
Element-wise Python functions can be used, too.'''
format = lambda x: '%.2f' % x
format(0.555)

frame.applymap(format)

'''
Series has a map method for applying an element-wise function:'''
series_e = frame['e']
series_e
series_e.map(format)


## 5.2.6 Sorting and ranking
'''
To sort lexicographically by row or by column index, use the sort_index method,
which returns a new, sorted object:'''
obj = pd.Series(range(4), index=list('dabc'))
obj

obj.sort_index()
obj.sort_values()

'''
With a DataFrame, you can sort by index on each axis.'''
frame = pd.DataFrame(np.arange(8).reshape(2,4), columns=list('dabc'),
                     index=['three','one'])
frame

frame.sort_index()
frame.sort_index(axis=1)


'''
The data is sorted in ascending order by default, but can be sorted
in descending order, too.'''
frame.sort_index(axis=1, ascending=False)


'''
To sort a Series by its values, use its sort_value method:'''
obj = pd.Series([4,7,-3,2])
obj
obj.sort_values()

'''
Any missing values are sorted to the end od the Series by default:'''
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])

obj.sort_values()


'''
On DataFrame, you may want to sort by the values in one or more columns.
To do so, pass one or more column names to the by option:'''
frame = pd.DataFrame({'b':[4,7,-3,2], 'a':[0,1,0,1]})
frame

frame.sort_values(by='b')

'''
To sort by multiple columns, pass a list of names:'''
frame.sort_values(by='a')
frame.sort_values(by=['a','b'])


'''
Ranking assigns ranks from one through the number of valid data points in 
an array. 
by default rank breaks ties by assigning each group the mean rank'''
obj = pd.Series([7,-5,7,4,2,0,4])
obj
obj.rank()

'''
Ranks can also be assigned according to the order they're observed in the data:'''
obj.rank(method='first')

'''
Naturally, you can rank in descending order, too:'''
obj.rank(ascending=False, method='max')

'''
DataFrame can compute ranks over the rows or the columns:'''
frame = pd.DataFrame({'b':[4.3, 7, -3, 2], 'a':[0,1,0,1],
                      'c':[-2,5,8,-2.5]})
frame
frame.rank(axis=1)


## 5.2.7 Axis indexes with duplicate values
'''
While many pandas functions (like reindex) require that the labels be unique,
it's not mandatory.'''
obj = pd.Series(range(5), index=['a','a','b','b','c'])
obj

obj.index.is_unique

'''
Indexing a value with multiple entries returns a Series while single entries
return a scalar value. And the same logic extends to indexing rows in a 
DataFrame:'''
obj['a']

df = pd.DataFrame(np.random.randn(4,3), columns=list('ccd'),
                  index=list('aabb'))
df

df['c']
df.loc['a']



# 5.3 Summarizing and Computing 
df = pd.DataFrame([[1.4, np.nan], 
                   [7.1, -4.5],
                   [np.nan, np.nan],
                   [0.75, -1.3]],
                   index=['a','b','c','d'],
                   columns=['one','two']) 
df
'''
Calling DataFrame's sum method returns a Series containing column sums:'''
df.sum()
'''
Passing axis=1 sums over the rows instead:'''
df.sum(axis=1)

'''
NA values are excluded unless the entire slice (rows or column) is NA. This
can be disabled using the skipna option:'''
df.sum(axis=1, skipna=False)


'''
Methods idxmin and idxmax return the index value where minimum or maximum values
are attained:'''
df

df.idxmax()
df.idxmax(axis=1)
df.idxmin()


'''
Other methods are accumulations.'''
df.cumsum()
df.cumsum(axis=1)


'''
method describe is producing multiple summary statistics in one shot:'''
df.describe()

'''
On non-numeric data, describe produces alternate summary statistics:'''
obj = pd.Series(['a','a','b','c']*3)
obj
obj.describe()

df
df.count()
df.max()
df.var()
df.std()

df.diff()
df.pct_change()


## 5.3.1 Correlation and Covariance
'''
Correlation and covariance are computed from pairs of arguments.
Let's consider some DataFrame of stock prices and volumnes obtained from
Yahoo!Finance:'''
import pandas_datareader as pdr

all_data = {}
for ticker in ['AAPL','IBM','MSFT','GOOG']:
    all_data[ticker] = pdr.get_data_yahoo(ticker)

price = pd.DataFrame({tic:data['Adj Close']
                    for tic, data in all_data.items()})

volume = pd.DataFrame({tic:data['Volume']
                     for tic, data in all_data.items()})

'''
Now compute percent changes of the prices:'''
returns = price.pct_change()

returns.tail()    

'''
The corr method of Series computes the correlation of the overlapping, non-NA,
aligned-by-index values in two Series. Relatedly, cov computes the covariance:'''
returns.AAPL.corr(returns.GOOG)

returns.AAPL.cov(returns.GOOG)

'''
DataFrame's corr and cov methods return a full correlation or covariance 
matrix as a DataFrame, respectively:'''
returns.corr()

returns.cov()

'''
Using DataFrame's corrwith method, you can compute pairwise correlations between
a DataFrame's columns or rows with another Series or DataFrame. 
Passing a Series returns a Series with the correlation value computed for 
each column:'''
returns.corrwith(returns.IBM)

'''
Passing a DataFrame computes the correlations of matching columns names.
Here, I compute correlations of percent changes with volumne:'''
returns.corrwith(volume)

'''
Passing axis=-1 does row-wise instead. 
In all cases, the data points are aligned by label before computing the 
correlation.'''

    
## 5.3.2 Unique values, value counts, and membership
'''
To extracts information about the values contained in a one-dimensional Series'''
obj = pd.Series(['c','a','d','a','a','b','b','c','c'])

unique = obj.unique()
unique
'''
The uniquo values are not necessarily returned in sorted order, but could be sorted
after the fact if needed (unique.sort()). Relateedly, value_counts computes a Series
containing value frequencies:'''
unique.sort()
unique

obj
obj.value_counts()

'''
value_counts is also available as a top-level pandas method that can be
used with any array or sequence:'''
pd.value_counts(obj.values, sort=False)

obj.value_counts(sort=False)

'''
isin is responsible for vectorized set membership and can be very useful
in filtering a data set down to a subset of values in a Series or column
in a DataFrame:'''
mask = obj.isin(['b','c'])
mask

obj[mask]
obj[(obj=='b')|(obj=='c')]

'''
To compute a histogram on multiple related columns in a DataFrame.'''
data = pd.DataFrame({'Qu1':[1,3,4,3,4],
                     'Qu2':[2,3,1,2,3],
                     'Qu3':[1,5,2,4,4]})
data

data.apply(pd.value_counts).fillna(0)    


# 5.4 Handling Missing Data
'''
pandas uses the floating point value NaN to represent missing data in both 
floating as well as non-floating point arrays.'''
import pandas as pd
import numpy as np

string_data = pd.Series(['aardvark','artichoke',np.nan,'avocdo'])
string_data

string_data.isnull()

'''
The built-in Python None value is also treated as NA in object arrays:'''
string_data[0] = None
string_data

string_data.isnull()
string_data.fillna(0)
string_data.dropna()
string_data


## 5.4.1 Filtering Out Missing Data
'''
dropna returns the Series with only the non-null data and index values:'''
data = pd.Series([1, None, 3.5, np.nan, 7])
data

data.dropna()
data
data[data.notnull()]

'''
With DataFrame objects, you may want to drop rows or columns which are all 
NA or just those containing any NAs. 
dropna by default drops any row containing a missing value:'''
data = pd.DataFrame([[1, 6.5, 3],
                     [1, None, None],
                     [None, None, None],
                     [None, 6.5, 3]])
data

data.dropna()    
data.dropna(axis=1)    

'''
Passing how='all' will only drop rows that are all NA:'''
data.dropna(how='all')    

'''
Dropping columns in the same way is only a matter of passing axis=1:'''
data[4] = None
data
data.dropna(axis=1, how='all')

'''
Suppose you want to keep only rows containing a certain number of observations.
You can indicate this with the thresh argument:'''
df = pd.DataFrame(np.random.randn(7,3))
df

df.iloc[:5,1]=None; 
df.loc[:4, 1]

df.iloc[:3,2] = None
df

df.dropna(thresh=2)
df.dropna()
df.dropna(how='all')
df

## 5.4.2 Filling in Missing Data
'''
Calling fillna with a constant replaces missing values with that value:'''
df = pd.DataFrame(np.random.randn(7,3))
df.iloc[:5,1]=None; df.iloc[:3,2] = None
df

df.fillna(0)
df

'''
Calling fillna with a dict you can use a different fill value for each column:'''
df.fillna({1:0.5, 2:-1})

'''
fillna returns a new object, but you can modify the existing object in place:'''
# always returns a reference to the filled object
df.fillna(0, inplace=True)
df

'''
The same interpolation methods available for reindexing can be used with fillna:'''
df = pd.DataFrame(np.random.randn(6,3))
df

df.iloc[2:,1] = None; df.iloc[4:,2]=None
df

df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)


'''
You can even pass the mean or median value of a Series:'''
data = pd.Series([1, None, 3.5, None, 7])
data

data.fillna(data.mean())



# 5.5 Hierarchical Indexing
'''
Hierachical indexing is an important feature of pandas enabling you to have
multiple (tow or more) index levels on an axis. 
It provides a way for you to work with higher dimensional data in a lower
dimensional form.'''


data = pd.Series(np.random.randn(10), 
                 index = [['a','a','a','b','b','b','c','c','d','d'],
                          [1,2,3,1,2,3,1,2,2,3]])
data    
data.index    
'''
The 'gaps' in the index display mean 'use the label directly above:'''

'''
with a hierarchically-indexed object, so called partial indexing is possible,
enabling you to concisely select subsets of the data:'''
data['b']

data['b':'c']
data.loc['b':'c']
data[['b','c']]

'''
Selection is even possible in some cases from an 'inner' level:'''
data[:,2]

'''
Hierarchical indexing plays a critical role in reshaping data and group-based
operatoins like forming a pivot table. 
For example, this data could be rearranged into a DataFrame using its 
unstack method:'''
data
data.unstack()
data.unstack().stack()

'''
With a DataFrame, either axis can have have a hierachical index:'''
frame = pd.DataFrame(np.arange(12).reshape(4,3),
                     index=[['a','a','b','b'],[1,2,1,2]],
                     columns=[['Ohio','Ohio','Colorado'],
                              ['Green','Red','Green']])
frame

'''
The hierarchical levels can have names (as string or any Python objects).'''
frame.index.name = ['key1','key2']
frame.columns.name=['state','color']
frame

frame['Ohio']

frame.unstack()
frame.stack()

## 5.5.1 Recording and Sorting Levels
'''
To rearrange the order of the levels on an axis or sort the data by the values
in one specific level.
The swaplevel takes two level numbers or names and returns a new object with
the levels interchanged (but the data is otherwise unaltered):'''
frame.swaplevel()
frame.swaplevel(axis=1)

'''
sortlevel sorts the data (stably) using only the values in a single level.
when swapping levels, it's not uncommon to also use sortlevel so that the 
result is lexicographically sorted:'''
frame.sortlevel()
frame.sortlevel(1)

frame.swaplevel().sortlevel()


## 5.5.2 Summary Statistics by Level
'''
You can specify the level you want to sum by an particular axis using a level
option:'''
frame

frame.sum()
frame.sum(level=0)
frame.sum(level=1)

frame.sum(axis=1)
frame.sum(axis=1).sum(level=0)
frame.sum(axis=1, level=1).sum(level=1)


## 5.5.3 Using a DataFrame's Columns
'''
To use one or more columns from a DataFrame as the row index;
Alternatively, to move the row index into the DataFrame's columns.'''

frame = pd.DataFrame({'a': range(7), 
                      'b': range(7,0,-1),
                      'c': ['one','one','one','two','two','two','two'],
                      'd': [0,1,2,0,1,2,3]})
frame

'''
DataFrame's set_index function will create a new DataFrame using one or 
more of its columns as the index:'''
frame2 = frame.set_index(['b','c'])
frame2
'''
By default the columns are removed from the DataFrame, though you can leave
them in:'''
frame.set_index(['c','d'], drop=False)

'''
reset_index does the opposite of set_index; the hierarchical index levels are
moved into the columns:'''
frame2
frame2.reset_index()
frame2.reset_index().reindex(columns=['a','b','c','d'])





# 5.6 Other pandas topics

## 5.6.1 Integer indexing
'''
In case where you need reliable position-based indexing regardless of the index
type, you can use the iget_value method from Series and irow and icol methods
from DataFrame:'''
ser3 = pd.Series(range(3), index=[-5, 1, 3])
'''
-5    0
 1    1
 3    2
dtype: int32'''

ser3.iget_value(2)

frame = pd.DataFrame(np.arange(6).reshape(3,2), index=[2,0,1])
frame

frame.irow(0)

## 5.6.1 Panel Data




