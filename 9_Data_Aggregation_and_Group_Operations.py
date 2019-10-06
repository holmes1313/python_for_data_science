

'''
1.Split a pandas object into pieces using one or more keys (in the form of functions,
arrays, or DataFrame column names)

2.Computing group summary statistics, like count, mean, or standard deviation, 
or a user-defined function

3.Apply a varying set of functions to each column of a DataFrame

4.Apply within-group transformations or other manipulations, like normalization,
linear regression, rank, or subset selection

5.Compute pivot tables and cross-tabulations

6.Perform quantitle analysis and other data-derived group values
'''

# 1.GroupBy Mechanics
'''
group operation: split - apply - combine
The splitting is performed on a particular axis of an object. For example, a 
DataFrame can be grouped on its rows (axis=0) or its columns (axis=1).

Each grouping key can take many forms, and the keys do not have to be all of the 
same type:
    * A list of array of values that is the same length as the axis being grouped
    * A value indicating a column name in a DataFrame
    * A dict or Series giving a correspondence between the values on the axis 
    being grouped and the group names
    * A function to be invoked on the axis index or the indivifual labels in 
    the index.
Note that the latter three methods are all just shortcuts for producing an array
of values to be used to split up the object.
'''

df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
df

'''
Suppose you wanted to compute the mean of the data1 column using the groups labels
from key1.
One way is to access data1 and call groupby with the column (a Series) at key1:'''
grouped = df['data1'].groupby(df['key1'])

'''
This grouped variable is a GroupBy object. This object has all of the information
needed to then apply some operation to each of the groups'''
grouped.mean()
'''
The data (a Series) has been aggregated according to the group key, producing a 
new Series that is now indexed by the unique values in the key1 column.

If instead we had passed multiple arrays as a list, we got something different:'''
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means

'''
In this case, we grouped the data using two keys, and the resulting Series
now has a hierarchical index consisting of the unique pairs of keys observed:'''

means.unstack()

'''
The group keys could be any arrays of the right length:'''
states = np.array(['Ohio','Ohio','California','California','Ohio'])

years = np.array([2005,2006,2005,2006,2005])

df['data1'].groupby([states, years]).mean()
df['data1'].groupby([states, years]).mean().unstack()


'''
Frequently the grouping information to be found in the same DataFrame as the 
data you want to work on. In that case, you can pass column names as the gorup
keys:'''
df.groupby('key1').mean()

'''
There is no key2 column in the result, because df['key2'] is not numeric data, 
it is said to be a nuisance column, which is therefore excluded from the result.
By default, all of the numeric columns are aggregated.'''

df.groupby(['key1','key2']).mean()


'''
Regardless of the objective in using groupby, a generally useful groupby method 
is size which returns a Series containting group sizes:'''
df

df.groupby(['key1','key2']).size()


## 1.1 Iterating Over Groups
'''
The GroupBy object supports iteration, generating a sequence of 2-tuples containing
the group name along with the chunk of data.'''
df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
df

df.groupby('key1')   # a Groupby object

for name, group in df.groupby('key1'):
    print(name)
    print(group)
    print()

list(df.groupby('key1'))


'''
In the case of multiple keys, the first element in the tuple will be a tuple of 
key values:'''
list(df.groupby(['key1','key2']))

for k1, group in df.groupby(['key1','key2']):
    print(k1)
    print(group)
    print()

for (k1, k2), group in df.groupby(['key1','key2']):
    print((k1,k2))
    print(k1)
    print(k2)
    print(group)
    print()


'''
A recipe you may find useful is computing a dict of the data pieces as a 
one-liner:'''

pieces = dict(list(df.groupby('key1')))
pieces

pieces['a']

dict(list(df.groupby(['key1','key2'])))[('a','one')]
df[df.key1=='a'][df.key2=='one']

'''
By deafult groupby groups on axis=0, but you can group on any of the other axes.'''
df.dtypes

grouped = df.groupby(df.dtypes, axis=1)
dict(list(grouped))


## 1.2 Selecting a Column or Subset of Columns
df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
'''
Indexing a Groupby object created from a DataFrame with a column name or array
of column names has the effect of selecting those columns for aggregation.
This means that:'''
df.groupby('key1')['data1']
df.groupby('key1')[['data2']]

'''
are syntactic sugar for:'''
df['data1'].groupby(df['key1'])
df[['data1']].groupby(df['key1'])

'''
Especially for large data sets, it may be desirable to aggregate only a few
columns.
e.g. to compute means for just the data2 column and get the result as a DataFrame'''
df.groupby([df['key1'],df['key2']])[['data2']].mean()  # a grouped DataFrame

# or
df[['data2']].groupby([df['key1'],df['key2']]).mean()

# which are different from
df.groupby([df['key1'],df['key2']])['data2'].mean()    # a grouped Series

df['data2'].groupby([df['key1'],df['key2']]).mean()


## 1.3 Grouping with Dicts and Series
'''
Grouping information may exist in a form other than an array'''
people = pd.DataFrame(np.random.randn(5,5),
                      columns=['a','b','c','d','e'],
                      index=['Joe','Steve','Wes','Jim','Travis'])
people

'''
loc works on labels in the index.
iloc works on the positions in the index (so it only takes integers).
ix usually tries to behave like loc but falls back to behaving like iloc 
if the label is not in the index.'''

people.iloc[2][['b','c']] = np.nan   # add a few na values
people

'''
Now, suppose I have a group of correspondence for the columns and want to sum
together the columns by group:'''
mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}

'''
Now, you could easily construct an array from this dict to pass to groupby,
but instead we can just pass the dict:'''
by_column = people.groupby(mapping, axis=1)
dict(list(by_column))

by_column.size()
by_column.sum()

'''
The same functionality holds for Series, which can be viewed as a fixed size
mapping.'''
map_series = pd.Series(mapping)
map_series

people.groupby(map_series, axis=1).count()


## 1.4 Grouping with Functions
'''
Using Python functions in what can be fairly creative ways is a more abstract
way of defining a group mapping compared with a dict or Series.
Any function passed as a group key will be called once per index value, with the
return values being used as the group names.

Suppose you want to group by the length of the names; you could compute an array
of string lengths, but instead you can just pass the len function:'''
people = pd.DataFrame(np.random.randn(5,5),
                      columns=['a','b','c','d','e'],
                      index=['Joe','Steve','Wes','Jim','Travis'])
people

people.groupby(len).sum()

'''
Mixing functions with arrays, dicts, or Series is not a problem as everything
gets converted to arrays internally:'''
key_list = ['one','one','one','two','two']

people.groupby([len, key_list]).min()


## 1.5 Grouping by Index Levels
'''
A final convenience for hierarchically-indexed data set is the ability to aggregate
using one of the levels of an axis index.
To do this, pass the level number or name using the level keyword:'''
hier_df = pd.DataFrame(np.random.randn(4,5),
                       columns=[['us','us','us','jp','jp'],[1,3,5,1,3]])
hier_df.columns.names =['city','tenor']                       
hier_df                       

hier_df.groupby(level='city',axis=1).count()
hier_df.groupby(level=0, axis=1).count()



# 2.Data Aggrevation
'''
Aggregation means any data transformation that produces scalar values from 
arrays.
You can use aggregations of your own devising and additionally call any method
that is also defined on the grouped object.'''
df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
df

'''
quantitle computes sample quantiles of a Series or a DataFrame's columns:'''
grouped = df.groupby('key1')

grouped['data1'].quantile(0.9)

'''
Initially, GroupBy efficiently slices up the Series, calls piece.quantile(0.9)
for each piece, then assembles those resultes together into the result object.'''

'''
To use your own aggregation functions, pass any function that aggregates an 
array to the aggregate or agg method:'''
def peak_to_peak(arr):
    return arr.max() - arr.min()

grouped.agg(peak_to_peak)

grouped[['data1','data2']].apply(peak_to_peak)

grouped.apply(peak_to_peak)
# TypeErroe

'''
Some methods like describe also work.'''
grouped.describe()


## 2.1 Column-wise and Multiple Function Application
tips = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_tips.csv')
tips.head()

'''Adding a tipping percentage column tip_pct'''
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.head()

# Column-wise and multiple function application
'''
As you've seen above, aggregating a Series or all of the columns of a DataFrame
is a matter of using aggregate with the desired function or calling a method like 
mean or std. However, you may want to aggregate using a different function 
depending on the column or multiple functions at once.

First, I'll group the tips by sex and smoker:'''
grouped = tips.groupby(['sex','smoker'])

'''
Note you can pasa the name of the statistics function as a string:'''
grouped_pct = grouped['tip_pct']

grouped_pct.mean()

grouped_pct.agg('mean')

grouped_pct.agg(np.mean)

grouped_pct.apply(np.mean)

'''
If you pass a list of functions or function names instead, you get back a DataFrame
with column names taken from the functionss:'''
def peak_to_peak(arr):
    return arr.max() - arr.min()

grouped_pct.agg(['mean','std',peak_to_peak])

'''
You don't need to accept the names that GroupBy gives to the columns.
You can pass a list of (name, function) tuples, the first element of each tuple
will be used as the DataFrame column names (you can think of a list of 2-tuples
as an ordered mapping):'''
grouped_pct.agg([('foo','mean'),('bar',np.std)])
                    
'''
With a DataFrame, you have more options as you can specify a list of functions
to apply to all of the columns or different functions per columns.
Suppose we wanted to compute the same three statistics for the tip_pct and 
total _bill columns:'''
functions = ['count', np.mean, np.max]

result = grouped[['tip_pct','total_bill']].agg(functions)
result

'''
As you can see, the resulting DataFrame has hierarchical columns, the same as you
would get aggregating each column separately and using concat to glue the results
together using the column names as the keys argument:'''

r1 = grouped['tip_pct'].agg(functions)
r2 = grouped['total_bill'].agg(functions)
pd.concat([r1,r2], axis=1,keys=['tip_pct','total_bill'])

result['tip_pct']

'''
As above, a list of tuples with custom names can be passes:'''
ftuples = [('Durchschnitt','mean'),('Abweichung',np.var)]

grouped[['tip_pct','total_bill']].agg(ftuples)


'''
Now, suppose you wantted to apply potentially different functions to one or more of 
the columns. 
The trick is to pass a dict to agg that contains a mapping of column names to 
any of the function specifications listed so far:'''
grouped.agg({'tip':np.max, 'size':'sum'})

grouped.agg({'tip_pct':['min','max','mean','std'],
             'size':'sum'})

'''
A DataFrame will have hierarchical columns only if multiple functions are applied
to at least one column.'''


## 2.2 Returning Aggregated Data in 'unindexed' Form
'''
In all examples up until now, the aggregated data comes back with an index, potentially
hierarchical, composed from the unique group key combinations observed. But you 
can disable this behavior in most cases by passing as_index=False to groupby:'''

tips = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.head()

tips.groupby(['sex','smoker']).mean()

tips.groupby(['sex','smoker'], as_index=False).mean()

'''
Of course, it's always possible to obtain the result in this format by calling
reset_index on the result.'''
tips.groupby(['sex','smoker']).mean().reset_index()


# 3.Group-wise Operations and Transformations
'''
Aggregation is only one kind of group operation. It accepts functions that 
reduces a one-dimensional array to a scalar value.
The transform and apply methods will enable you to do many other kinds of groups 
opertions.

Suppose we wannted to add a column to a DataFrame containing group means for 
each index. 
One way to do this is to aggregate, then merge:'''
df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
df

k1_means = df.groupby('key1').mean().add_prefix('mean_')

pd.merge(df, k1_means, left_on='key1', right_index=True)

'''
This works, but is somewhat inflexible. You can think of the opertion as transforming
the two data columns using the np.mean function.
Let's use the transform method on GroupBy:'''
people = pd.DataFrame(np.random.randn(5,5),
                      columns=['a','b','c','d','e'],
                      index=['Joe','Steve','Wes','Jim','Travis'])
people

key=['one','two','one','two','one']

people.groupby(key).mean()

people.groupby(key).transform(np.mean)

'''
Transform applies a function to each group, then places the results in the appropriate
locations. If each group produces a scalar value, it will be prpopagated (broadcasted).

Suppose instead you wannted to subtract the mean value from each group.
To do this, create a demeaning function and pass it to transform:'''
def demean(arr):
    return arr - arr.mean()

demeaned = people.groupby(key).transform(demean)
demeaned

'''
transform is a more specialized function having rigid requirements: the passed
function must either produce a scalar value to be broadcasted or a transformed
array of the same size.'''


## 3.1 Apply: General split-apply-combine
'''
The most general purpose GroupBy method is apply. Apply splits the object being 
manipulated into pieces, invokes the passed function on each piece, then attempts 
to concatenate the pieces together.'''
tips = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:6]

'''
Suppose you wanted to select the top five tip_pct values by group.
First, it's straightforward to write a function that selects the rows with the largest
values in a particular column:'''
a = pd.DataFrame({'a':[2,3,4,2,1,4,5],
                 'b':[324,4534,232,4534,23,453,3]})
a.sort_values(by='a', ascending=False)[:5]

def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column, ascending=False)[:n]
    
top(tips,6)

'''
Now, if we group by smoker, say, and call apply with this function:'''
tips.groupby('smoker').apply(top)

tips.groupby('smoker').agg('count')

'''
What happened here?
The top functon is called on each piece of the DataFrame, then the results are 
glued using pandas.concat, labeling the pieces with the group names. The result
therefore has a hierarchical index whose inner level contains index values from 
the original DataFrame. 

If you pass a function to apply that takes other arguments or keywords, 
you can pass these after the function:'''
tips.groupby(['smoker','day']).apply(top,n=1,column='total_bill')

'''
You may recall above I called describe on a GroupBy object:'''
result = tips.groupby('smoker')['tip_pct'].describe()
result

result.unstack().unstack()


'''
Inside Groupby, when you invoke a method like describe, it is actually just a 
shortcut for:'''
f = lambda x : x.describe()
tips.groupby('smoker')['tip_pct'].apply(f).unstack('smoker')

# Suppressing the group keys'''
tips.groupby('smoker',group_keys=False).apply(top)



## 3.2 Quantile and Bucket Analysis
'''
Pandas has some tools, in particular cut and qcut, for slicing data up into buckets
with bins of your choosing or by sample quantiles.
Combining these fucntions with groupby, it becomes very simple to perform 
bucket or quantitle analysis on a data set.

Consider a simple random data set and an equal-length bucket categorization
using cut:'''
frame = pd.DataFrame({'data1':np.random.randn(100),
                      'data2':np.random.randn(100)})
frame[:10]

factor = pd.cut(frame.data1, 4, labels=False)
factor[:10]

'''
The factor object returned by cut can be passed directly to groupby.
So we could compute a set of statistics for the data2 column like so:'''
def get_stats(group):
    return {'min':group.min(), 'max':group.max(),
            'count': group.count(), 'mean':group.mean()}

grouped = frame.data2.groupby(factor)

grouped.apply(get_stats).unstack()

'''
These were equal-length buckets; to compute equal-size buckets based on sample
quantiles, use qcut. I'll pass labels=False to jsut get quantile numbers.'''
# return quantile numbers
grouping = pd.qcut(frame.data1, 10, labels=False)

grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()



grouping = pd.qcut(frame.data1, 10, labels=False)

grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()



## 3.3 Example: Filling Missing values with Group-specific Values
'''
When cleaning up missing data, in some cases you will filter out data obervations
using dropna:'''
s = pd.Series(np.random.randn(6))
s
s[::2] = np.nan
s.dropna().reset_index()[0]
s

'''
but in others you may want to impute (fill in) the NA values using 
a fixed value or some value drived from the data. fillna is the right tool to use;
for example here I fill in Na values with the mean:'''
s = pd.Series(np.random.randn(6))
s
s[::2] = np.nan
s

s.fillna(s.mean())

'''
Suppose you need to fill value to vary by group. As you may guess, you need only 
group the data and use apply with a function that calls fillna on each data chunk.'''
states = ['Ohio','New York','Vermont','Florida',
          'Oregon', 'Nevada','California','Idaho']

group_key = ['East']*4 + ['West']*4

data = pd.Series(np.random.randn(8), index=states)
data

data[['Vermont','Nevada','Idaho']] = np.nan
data

data.groupby(group_key).mean()

'''
We can fill the NA values using the group mean like so:'''
fill_mean = lambda g: g.fillna(g.mean())

data.groupby(group_key).apply(fill_mean)

'''
In another case, you might have pre-defined fill values in your code that vary by
group. Since the groups have a name attribute set internally, we can use that:'''
fill_values = {'East':0.5, 'West':-1}

fill_func = lambda g: g.fillna(fill_values[g.name])

data.groupby(group_key).apply(fill_func)

for name, data in data.groupby(group_key):
    print(name)
    print(data)
    print()


## 3.4 EXample: Random Sampling and Permutation
'''
Suppose you wanted to draw a frandom sample (with or without replacement) from
a large dataset, there are a number of ways to perform the 'draws';
One way is to select the first k elements of np.random.permutation(N),
where N is the size of your complete dataset and K the desited sample size.
Here's a way to construct a deck of English-style playing cards:'''
# Hearts, Spades, Clubs Dimonds
suits = ['H','S','C','D']
card_val = (list(range(1,11))+[11]*3)*4
base_names = ['A']+list(range(2,11)) + ['J','Q','K']

cards = [suit + str(base_name) for suit in suits for base_name in base_names]

deck = pd.Series(card_val, index=cards)


# append() and extend(): 
'''
The former appends an object to the end of the list (e.g., another list) 
while the later appends each element of the iterable object (e.g., anothre list) 
to the end of the list.'''
cards = []
aa = [1]

cards.extend(aa)
cards

cards.append(aa)
cards
########

# permutation: sampling without replacement
# select k(5) out of N(10)
import numpy as np
np.random.permutation(10)[:5]


'''
Drawing a hand of 5 cards fromt the desk could be written as:'''
def draw(deck, n=5):
    return deck.get(np.random.permutation(len(deck))[:n])

draw(deck)

'''
Suppose you wanted two random cards from each suit. Becasue the suit is the last
character of each card name, we can group based on this and use apply:'''
deck
get_suit = lambda card: card[0]  # first letter is suit

deck.groupby(get_suit).apply(draw, n=2)

# alternatively
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)


## 3.5 Example: Group Weighted Average and Correlation
'''
Under the split-apply-combine paradigm of groupby, operations between columns 
in a DataFrae or two Series, such a group weighted average, become a routine 
affair. As an example, take this dataset containing group keys, values, and
some weights:'''
df = pd.DataFrame({'category':['a']*4+['b']*4,
                   'data':np.random.randn(8),
                   'weights': np.random.rand(8)})
df

# weight average
np.average(df['data'],weights=df['weights'])

np.average(df['data'])        

# the group weight average by category        
grouped = df.groupby('category')

get_wavg = lambda g: np.average(g['data'], weights=g['weights'])        

grouped.apply(get_wavg)        


'''
As a less trivial example, consider a data set from Yahoo! Finance containing 
end of day prices for a few strocks and the S&P 500 index (the SPX ticker):'''

close_px = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science'\
                       '\ch09_stock_px.csv', parse_dates=True, index_col=[0])

close_px.info()

close_px[-4:]

'''
One task of interest might be to compute a DataFrame consisting of the yearly
correlation of daily reutrns (computed from percent changes) with SPX.'''

# daily return percent change
rets = close_px.pct_change().dropna()

spx_corr = lambda g: g.corrwith(g['SPX'])

by_year = close_px.groupby(lambda g: g.year)

by_year.apply(spx_corr)

'''
This is, of course, nothing to stop you from computing iner-column correlations:'''
# annual correlation of apple with microsoft
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))


## 3.6 Example: Group-wise Linear Regression
'''
You can use groupby to perform more complex group-wise statistics analysis, as long
as the function returns a pandas object or scalar value.
For example, I can define the following regress function (using the statsmodels
econometrics library) which executes an ordinary least squares (OLS) regression 
on each chunk of data:'''
import statsmodels.api as sm

def regress(data, yvar, xvar):
    Y = data[yvar]
    X = data[xvar]
    X['intervept'] = 1
    result = sm.OLS(Y,X).fit()
    return result.params

regress(close_px, 'AAPL', ['MSFT'])

by_year.apply(regress, 'AAPL',['MSFT'])


# 4.Pivot Tables and Cross-Tabulation
'''
Pivot table aggregates a table of data by one or more keys, arranging the data 
in a rectangle with some of the group keys along the rows and some along 
the columns.
Returning to the tipping data set, suppose I wanted to compute a table of 
group means (the default pivot table aggregation type) arranged by sex and 
smoker on the rows:'''
tips = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_tips.csv')
tips.head()

tips.pivot_table(index=['sex','smoker'], aggfunc='mean')

'''
This could have been easily produced using groupby.'''
tips.groupby(['sex','smoker']).apply(np.mean)

'''
Now, suppose we want to aggregate only tip_pct and size, and additionally 
group by day. I will put smoker in the table columns and day in the rows:'''
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.head()

tips.pivot_table(['tip_pct','size'], index=['sex','day'], 
                 columns='smoker', aggfunc='mean')

'''
This table could be augmented to include partial totals by passing margins=True.
This has the effect of add All row and column labels, with corresponding values
being the group statistics for all the data within a single tier.

In this below example, the All values are means without taking into account 
smoker vs. non-smoker (the All columns) or any of the two levels of grouping
on the rows (the All row):'''
tips.pivot_table(['tip_pct','size'], index=['sex','day'],
                 columns='smoker',aggfunc='mean', margins=True)

tips.groupby('smoker')['size'].apply(np.mean)

'''
To use a different aggregation function, pass it to aggfunc.
For example, 'count' or 'len' will give you a cross-tabulation (count or frequency)
of group sizes:'''
tips.pivot_table('tip_pct',index=['sex','smoker'],columns='day',
                 aggfunc=len, margins=True)

tips.pivot_table('tip',index=['sex','smoker'],columns='day',
                 aggfunc='count', margins=True)

'''
If some combinations are empty (or otherwise NA), you may wish to pass a 
fill_value:'''
tips.pivot_table('size',index=['time','sex','smoker'],
                 columns='day',aggfunc='sum')

tips.pivot_table('size',index=['time','sex','smoker'],
                 columns='day',aggfunc='sum', fill_value=0)


## 4.1 Cross-Tabulations: Crosstab
'''
A cross-tabulation (or crosstab for short) is a special case of a pivot table 
taht computes group frequencies.'''
data = pd.DataFrame({'Sample':list(range(1,11)),
                     'Gender':['F','M','F','M','M','M','F','F','M','F'],
                     'Handedness':['r','l','r','r','l','r','r','r','r','r']})
data

'''
We might want to summarize this data by gender and handedness. 
The pd.crosstab function is very convenient:'''
pd.crosstab(data.Gender, data.Handedness, margins=True)

pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)

# or you can use pivot_table
data.pivot_table(index='Gender',columns='Handedness',aggfunc=len,
                 margins=True, fill_value=0)

tips.pivot_table('total_bill',index=['time','day'], columns='smoker',
                 aggfunc=len, margins=True, fill_value=0)



## Example: 2012 Federal Election Commission DataBase
'''
The US Federal Election Commission pyblishes data on contributions on political
campaigns. This includes contributor names, occupation and employer, address, 
and contribution amount.'''
import pandas as pd
fec = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                  '\ch09_P00000001-ALL.csv')
fec.info()

fec.iloc[123456]

'''
You can see that there are no political party affiliations in the data, so this
would be useful to add.
You can get a list of all the unique political candidates using unique 
(note that NumPy suppresses the quotes around the strings in the output):'''
unique_cands = fec.cand_nm.unique()
unique_cands

len(unique_cands)
unique_cands[2]

'''
An easy way to indicate party affiliation is using a dict:'''
parties = {'Bachmann, Michelle':'Republican',
           'Romney, Mitt':'Republican',
           'Obama, Barack':'Democrat',
           "Roemer, Charles E. 'Buddy' III":'Republican',
           'Pawlenty, Timothy':'Republican',
           'Johnson, Gary Earl':'Republican',
           'Paul, Ron':'Republican',
           'Santorum, Rick':'Republican',
           'Cain, Herman':'Republican',
           'Gingrich, Newt':'Republican',
           'McCotter, Thaddeus G':'Republican',
           'Huntsman, Jon':'Republican',
           'Perry, Rick':'Republican'}

'''
Now using this mapping and the map method on Series object, you can compute 
an array of political parties from the candidate names:'''
fec.cand_nm[:5].map(parties)

# Ad it as a column
fec['party'] = fec.cand_nm.map(parties)

fec['party'].value_counts()

# value_counts() method test
aaa = pd.DataFrame({'a':[1,1,1,1,1],
                    'b':['a','b','c','d','e'],
                    'c':[1,2,34,4,5]})
aaa
aaa['a'].value_counts()
aaa['b'].value_counts()
aaa['c'].value_counts()
aaa['a'].sum()
aaa['b'].sum()
aaa['c'].sum()


'''
A couple of data preparation points. 
First, this data includes both contributions and refunds (negative 
contribution amount):'''                                              
(fec.contb_receipt_amt > 0).value_counts()

(fec.receipt_desc == 'Refund').value_counts()

fec[fec.contb_receipt_amt<0]['receipt_desc'].value_counts()

'''
To simpligy the analysis, I'll restrict the data set to positive contributions:'''
fec = fec[fec.contb_receipt_amt>0]

(fec.contb_receipt_amt>0).value_counts()

'''
Since Obama and Romney are the main two candidates, I'll also prepare a subset that 
just has contributions to their campaigns:'''
fec.cand_nm.value_counts()

fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
fec_mrbo.cand_nm.value_counts()


# Donation Statistics by Occipation and Employer
'''
Donations by occupation is another oft-studied statistics.
First, the total number of donaions by occupatinon is easy:'''
fec.contbr_occupation.value_counts()[:10]

'''
You will notice by looking at the occupations that many refer to the same basic
job type, or there are several variants of  the same thing.
Here is a code snippet illustrates a technique for cleaning up a few of them by 
mapping from one occupations to another; 
note the "trick" of using dict.get to allow occupations with no mapping to 
"pass through":'''

occ_mapping = {
        'INFORMATION REQUESTED': 'NOT PROVIDED',
        'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
        'INFORMATION REQUESTED (BEST EFFORTS)': 'NOR PROVIDED',
        'C.E.O.': 'CEO'}

# If no mapping provided, return x
f = lambda x: occ_mapping.get(x,x)

occ_mapping.get('retired','retired')

fec.contbr_occupation = fec.contbr_occupation.map(f)

fec.contbr_occupation.value_counts()[:10]

'''
I'll also do the same thing for employers:'''
fec.contbr_employer.value_counts()[:10]

emp_mapping = {
        'INFORMATION REQUESTED': 'NOT PROVIDED',
        'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
        'SELF': 'SELF-EMPLOYED',
        'SELF EMPLOYED': 'SELF-EMPLOYED',
        'UNEMPLOYED': 'NOT-EMPLOYED',
        'NOT EMPLOYED': 'NOT-EMPLOYED',
        'NOT PROVIDED': 'NOT-PROVIDED',
        'NONE': 'NOT-PROVIDED'}

# If not mapping provided, return x
e = lambda x: emp_mapping.get(x,x)

fec.contbr_employer = fec.contbr_employer.map(e)

fec.contbr_employer.value_counts()[:10]


'''
Now, you can use pivot_table to aggregate the data by party and occupation,
then filter down to the subset that donated at least $2 million overall:'''

by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party', aggfunc='sum')

over_2m = by_occupation[by_occupation.sum(axis=1)>2000000]
over_2m

'''
It can be easier to look at this data graphically as a bar plot:'''
over_2m.plot(kind='barh')


'''
You might be interestd in the top donor occupations or top companies donating to
Obama and Romney.
To do this you can group by candidate name and use a variant of the top method
from earlier in the chapter:'''

def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    
    # Order totals by key in descending order
    return totals.sort_values(ascending=False)[:n]
    
'''
The aggregated by occupation and employer:'''
grouped = fec_mrbo.groupby('cand_nm')

grouped.apply(get_top_amounts, 'contbr_occupation', n=7)

grouped.apply(get_top_amounts, 'contbr_employer', n=10)

# test. why different result?
grouped2 = fec_mrbo.groupby(['cand_nm', 'contbr_occupation'])
aa = grouped2['contb_receipt_amt'].sum()
type(aa)
aa 
o = aa['Obama, Barack'].sort_values(ascending=False)[:10]
r = aa['Romney, Mitt'].sort_values(ascending=False)[:10]
o.index = list(zip(['Obama']*len(o.index),o.index))


# Bucketing Donation Amounts
'''
A useful way to analyze this data is to use the cut function to discretize the
contributor amounts into buckets by contribution size:'''
import numpy as np
bins = np.array([0,1,10,100,1000,10000,100000,1000000,10000000])

labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
labels

'''
We can then group the data for Obama and Romney be name and bin label to get 
a histogram by donation size:'''
grouped = fec_mrbo.groupby(['cand_nm', labels])

grouped.size().unstack(0)
'''
This data shows that Obama has received a significently larger number of small 
donations than Romney. 
You can also sum the contribution amounts and normalize within buckets to 
visualize percenage of total donations of each size by candidate:'''
bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
bucket_sums

normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
normed_sums

normed_sums[:-2].plot(kind='barh',stacked=True)

'''
I excluded the two largest bins as these are not donations from individuals.'''

# self analysis by aggreating donations by state
grouped2 = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
aa = grouped2.contb_receipt_amt.sum().unstack(0).fillna(0)
aa['sum'] = aa['Obama, Barack'] + aa['Romney, Mitt']
aa
bb = aa.sort_values(by='sum',ascending=False)[:10]
bb
bb['Obama_pct'] = bb['Obama, Barack'] / bb['sum']
bb['Romney_pct'] = bb['Romney, Mitt'] / bb['sum']
bb
bb[['Obama_pct', 'Romney_pct']].plot(kind='barh',stacked=True)


# Donation Statistics by State
'''
Aggregating the data by candidate and state is a routine affair:'''

grouped = fec_mrbo.groupby(['cand_nm','contbr_st'])

totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)

totals = totals[totals.sum(1)>100000]

totals[:10]

'''
If you divide each row by the total contribution amount, you get the relative 
percentage of total donations by state for each candidate:'''

percent = totals.div(totals.sum(1), axis=0)

percent.plot(kind='barh', stacked=True)

