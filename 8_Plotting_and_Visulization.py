
# 1.A Brief matplotlib API Primer

import matplotlib.pyplot as plt

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)

fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

import numpy as np

plt.plot(np.random.randn(50).cumsum(), 'k--')

ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)

ax2.scatter(np.arange(30), np.arange(30)+3*np.random.randn(30))

fig, axes = plt.subplots(2,3)
fig, axes = plt.subplots(1,1)
axes

fig, axes = plt.subplots(2,2,sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)


plt.plot(np.random.randn(30).cumsum(), 'ko--')
plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='--', marker='o')

data = np.random.randn(30).cumsum()
data

fig = plt.figure(); ax = fig.add_subplot(1,1,1)
plt.plot(data, 'k--')
plt.plot(data, 'k-', drawstyle='steps-post')


fig=plt.figure(); ax=fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())

ticks = ax.set_xticks([0,250,500,750,1000])
labels = ax.set_xticklabels(['one','two','three','four','five'],
                            rotation=30, fontsize='large')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')


import matplotlib.pyplot as plt
fig = plt.figure(); ax=fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')
ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
ax.legend(loc='best')

ax.text(200, 20, 'Hello world!', family='monospace', fontsize=10)


import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
fig=plt.figure(); ax=fig.add_subplot(1,1,1)

data = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_spx.csv', index_col = 0, parse_dates = True)
data.head()
spx = data['SPX']
spx.head()
type(spx.index)
spx.plot(style='k-')

crisis_data = [
        (datetime(2007,10,11), 'Peak of bull market'),
        (datetime(2008,3,12), 'Bear Stearns Fails'),
        (datetime(2008,9,15),'Lehman Bankruptcy')]

date = datetime(2007,10,11)
spx.asof(date)
spx[date]


ax.annotate('Peak of bull market', xy=(datetime(2007,10,11), spx[datetime(2007,10,11)]+50),
            xytext=(datetime(2007,10,11), spx[datetime(2007,10,11)]+200),
            arrowprops=dict(facecolor='black'))


# Line plots
fig, axes = plt.subplots(1,1)
s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0,100,10))
s.plot(use_index=False, xticks=[0,5,10], xlim=[0,11])

df = pd.DataFrame(np.random.randn(10,4).cumsum(0))
df
df.plot(xlim=[0,10])

pd.DataFrame([[1,2,3],[4,5,6]]).cumsum(0)


# Bar plots
fig, axes = plt.subplots(2,1)
data = pd.Series(np.random.rand(16),index=list('abcdefghijklmnop'))
data.plot(kind='bar', ax=axes[0])
data.plot(kind='barh', ax=axes[1])

df = pd.DataFrame(np.random.rand(6,4),
                  index=['one','two','three','four','five','six'],
                  columns=pd.Index(['A','B','C','D'], name='Genus'))
df

df.plot(kind='bar')

df.plot(kind='bar',stacked=True)

df.plot(kind='barh',stacked=True)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tips = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_tips.csv')
tips.head()
tips['size']
party_counts = pd.crosstab(tips['day'], tips['size'])
party_counts = party_counts.iloc[:, 1:-1]

party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0)
party_pcts.plot(kind='bar', stacked=True)


# Histogram and Density Plots
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips['tip_pct'].hist(bins=50)

tips['tip_pct'].plot(kind='kde')


comp1 = np.random.normal(0,1,200)
comp2 = np.random.normal(10,2,200)  # N(10,4)
values = pd.Series(np.concatenate([comp1, comp2]))
values.hist(bins=100, alpha=0.3, color='k', normed=True)
values.plot(kind='kde', style='k--')


# Scatter plots
macro = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                    '\macrodata.txt')
macro.head()

data = macro[['cpi','m1','tbilrate','unemp']]
data.head()

data.head()
trans_data = np.log(data).diff().dropna()

plt.scatter(trans_data['m1'], trans_data['unemp'])
plt.title('Changes in log %s vs. log %s' % ('m1','unemp'))

pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)


#  Plotting map

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_Haiti.csv')
data.info()
data.head()
data.shape
data.columns

data[['INCIDENT DATE', 'LATITUDE','LONGITUDE']][:10]
data = data[(data['LATITUDE']>18) & ()]


data[['CATEGORY']].head()



















%matplotlib

## 1.1 Figures and Subplots
'''
Plots in matplotlib reside within a Figure object.'''
fig = plt.figure()      # create a new figure

'''
You can’t make a plot with a blank figure. You have to create one or more 
subplots using add_subplot:'''
ax1 = fig.add_subplot(2,2,1)

'''
This means that the figure should be 2 × 2, and we’re selecting the first of 
4 subplots (numbered from 1).'''
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)

'''
When you issue a plotting command like plt.plot([1.5, 3.5, -2, 1.6]), matplotlib
draws on the last figure and subplot used (creating one if necessary), 
thus hiding the figure and subplot creation.'''
import numpy as np

plt.plot(np.random.randn(50).cumsum(), 'k--')
''''
The 'k--' is a style option instructing matplotlib to plot a black dashed line. 

The objects returned by fig.add_subplot above are AxesSubplot objects, 
on which you can directly plot on the other empty subplots by calling each 
one’s instance methods'''

ax1.hist(np.random.randn(100),bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30)+3*np.random.randn(30))

'''
Since creating a figure with multiple subplots according to a particular layout 
is such a common task, there is a convenience method, plt.subplots, that creates 
a new figure and returns a NumPy array containing the created subplot objects:'''
fig, axes = plt.subplots(2,3)

axes[1,1].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)

'''
This is very useful as the axes array can be easily indexed like a two-dimensional 
array. You can also indicate that subplots should have the same X
or Y axis using sharex and sharey, respectively. This is especially useful when 
comparing data on the same scale; otherwise, matplotlib auto-scales plot 
limits independently.'''


### Adjusting the spacing around subplots
'''
The spacing can be most easily changed using the subplots_adjust Figure method, 
also available as a top-level function:

subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, 
hspace=None)

wspace and hspace controls the percent of the figure width and figure height, 
respectively, to use as spacing between subplots. 

Here is a small example where I shrink the spacing all the way to zero
'''
fig, axes = plt.subplots(2,2,sharex=True, sharey=True)

for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)

plt.subplots_adjust(wspace=0, hspace=0)


## 1.2 Colors, Markers, and Line Styles
'''
Matplotlib’s main plot function accepts arrays of X and Y coordinates and 
optionally a string abbreviation indicating color and line style.

For example, to plot x versus y with green dashes, you would execute:
    ax.plot(x, y, 'g--')

The same plot could also have been expressed more explicitly as:
    ax.plot(x, y, linestyle='--', color='g')

Line plots can additionally have markers to highlight the actual data points.
The marker can be part of the style string, which must have color followed by marker type and line style
'''
plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='dashed', marker='o')

plt.plot(np.random.randn(30).cumsum(), 'ko--')

'''
For line plots, you will notice that subsequent points are linearly interpolated 
by default. This can be altered with the drawstyle option:'''
data = np.random.randn(30).cumsum()

plt.plot(data, 'k--', label='Default')

plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')

plt.legend(loc='best')


## 1.3 Ticks, Labels, and Legends
'''
xlim, xticks, and xticklabels control the plot range, tick locations, and tick 
labels, respectively.
They can be used in two ways:
    1.Called with no arguments returns the current parameter value.For example
plt.xlim() returns the current X axis plotting range.
    2.Called with parameters sets the parameter value. So plt.xlim([0, 10]), sets 
    the X axis range to 0 to 10.
    
All such methods act on the active or most recently-created AxesSubplot. 
Each of them corresponds to two methods on the subplot object itself; 
in the case of xlim these are ax.get_xlim and ax.set_xlim.'''

### Setting the title, axis labels, ticks, and ticklabels
'''
To illustrate customizing the axes, I’ll create a simple figure and plot of 
a random walk'''
fig = plt.figure(); ax = fig.add_subplot(1,1,1)

# or
fig, ax1 = plt.subplots(1,1)

ax.plot(np.random.randn(1000).cumsum())

'''
To change the X axis ticks, it’s easiest to use set_xticks and set_xticklabels. The
former instructs matplotlib where to place the ticks along the data range; by default
these locations will also be the labels. But we can set any other values as the labels using
set_xticklabels:'''
ticks = ax.set_xticks([0,250,500,750,1000])
ticks

labels = ax.set_xticklabels(['one','two','three','four','five'],
                            rotation=30, fontsize='x-large')
labels

'''
Lastly, set_xlabel gives a name to the X axis and set_title the subplot title:'''
ax.set_title('My first matplotlib plot')

ax.set_xlabel('Stages', fontsize='xx-large')

'''
Modifying the Y axis consists of the same process, substituting y for x in above.'''
ax.set_yticks([-10, 0, 10 ,20, 30])

ax.set_yticklabels(['first','second','third','forth','fifth'])

ax.set_ylabel('y label', fontsize='xx-large')

### Adding legends
'''
Legends are another critical element for identifying plot elements. 
There are a couple of ways to add one. The easiest is to pass the label argument 
when adding each piece of the plot:'''

fig = plt.figure() ; ax = fig.add_subplot(1,1,1)

# or
fig, ax2 = plt.subplots(1,1)

ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')

ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')

ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')

'''
Once you’ve done this, you can either call ax.legend() or plt.legend() to automatically
create a legend:'''
ax.legend(loc='best')

'''
The loc tells matplotlib where to place the plot. If you aren’t picky
'best' is a good option, as it will choose a location that is most out of the way. 
To exclude one or more elements from the legend, pass no label or label='_nolegend_'.'''


## 1.4 Annotations and Drawing on a Subplot
'''
Annotations can draw both text adn arrows arranged appropriately. As an example,
let's plot the closing S&P 500 index price since 2007 and annotate it with some
of the important dates from the 2008-2009 financial crisis.'''   
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib
import pandas as pd

fig = plt.figure(); ax = fig.add_subplot(1,1,1)

data = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_spx.csv', index_col = 0, parse_dates = True)
data.head()
spx = data['SPX']
type(spx)

ax.plot(spx, 'k-')

# or  spx.plot(style='k-')

crisis_data = [
        (datetime(2007,10,11), 'Peak of bull market'),
        (datetime(2008,3,12), 'Bear Stearns Fails'),
        (datetime(2008,9,15), 'Lehman Bankruptcy')
]

'''
TEST:
ax.annotate('peak of bull market', xy=(datetime(2007,10,11), spx.asof(datetime(2007,10,11))+50),
            xytext=(datetime(2007,10,11), spx.asof(datetime(2007,10,11))+200),
            arrowprops=dict(facecolor='red'))'''

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date)+50),
                xytext=(date, spx.asof(date)+200),
                arrowprops=dict(facecolor='red'))

# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007','1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in 2008-2009 financial crisis')


'''
Drawing shapes requires some more cares. matplotlib has objects that represent
many common shapes, referred to as patches.
To add a shape t a plot, you create the patch object shp and add it to 
a subplot by calling ax.add_patch(shp):'''
fig, ax = plt.subplots(1,1)

rect = plt.Rectangle((0.2,0.75), 0.4, 0.15, color='k',alpha=0.3)

circ = plt.Circle((0.7,0.2),0.15, color='b', alpha=0.3)

pgon = plt.Polygon([[0.15,0.15], [0.35,0.4], [0.2,0.6]], color='g',alpha=0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)


## 1.5 Saving Plots to File
'''
The active figure can be saved to file using plt.savefig:
    
    plt.savefig('figpath.svg')
The file type is inferred from the file extension. 
There are a couple of important options: dpi, which controls the dots-per-inch
resolution, and bbox_inches, which can trim the whitespace around the actual figure:
    
    plt.savefig('figpath.png', dpi=400, bbox_inches='tight')

savefig doesn't have to write to disk; it can also write to any file-like object,
such as a BytesIO:
    from io import BytesIO
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()'''

## 1.6 matplotlib Configuration



# 2.Plotting Functions in pandas
## 2.1 Line Plots
'''
Series and DataFrame each have a plot method for making many different plot 
types. By default, they make line plots:'''
s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0,100,10))
fig, ax = plt.subplots(1,1)
fig1, ax1 = plt.subplots(1,1)

s.plot(style='ko--', ax=ax)

'''
The Series object's index is passed to matplotlib for plotting on the X axis,
though this can be disabled by passing use_index=False.
The X axis ticks and limits can be adjusted using the xticks and xlim options.'''

'''
DataFrame's plot method plots each of its columns as a different line on the 
same subplot, creating a legend automatically:'''
df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
                  columns=['A','B','C','D'],
                  index=np.arange(0,100,10))
df
df.plot(subplots=True, sharey=True, sharex=True, sort_columns=True)


## 2.2 Bar Plots
'''
Making bar plots instead of line plots is as simple as passing kind='bar' or 
kind='barh'.
In this case, the Series or DataFrame index will be used as the X(bar) or Y(barh)
ticks.'''
fig, axes = plt.subplots(2,1)

data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data

data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)
data.plot(kind='barh', ax=axes[1], color='r', alpha=0.7)

data.plot(kind='bar', ax=axes[0], color='k',alpha=0.7)
data.plot(kind='barh',ax=axes[1], color='k',alpha=0.7)

'''
With a DataFrame, bar plots group the values in each row together in a group
in bars, side by side, for each value.'''

df = pd.DataFrame(np.random.rand(6,4),
                  index=['one','two','three','four','five','six'],
                  columns=[pd.Index(['A','B','C','D'], name='Genus')])
df

df.plot(kind='barh')
'''
Note that the name 'Genus' on the DataFrame's columns is used to title the legend.

Stacked  bar plots are created from a DataFrame by passing stacked=True, resulting
in the value in each row being stacked together.'''
df.plot(kind='barh', stacked=True, alpha=0.5)

'''
Returning to the tipping data set used earlier in the book, suppose we wanted to
make a stacked bar plot showing the percentage of data points for each party 
size on each day. I load the data using read_csv and make a cross-tabulation by 
day and party size:'''
tips = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_tips.csv')
tips.head()
tips.shape

party_counts = pd.crosstab(tips['day'], tips['size'])
party_counts

# Not many 1- and 6-person parties
party_counts = party_counts.loc[:,2:5]
party_counts

'''
Then, normalize so that each row sums to 1 and make the plot:'''
# Normailze to sum to 1
party_counts.plot(kind='bar', stacked=True)

party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0)
party_pcts

party_pcts.plot(kind='bar', stacked=True)

'''
so you can see that party sizes appear to increase on the weekendin this data
set.'''


## 2.3 Histogram and Density Plots
'''
a histogram is a kind of bar plot that gives a discretized display of value 
frequency. The data points are split into discrete, evenly spaced bins, and the 
number of data points in each bin is plotted.

Using the tipping data from before, we can make a histogram of tip percentages
of the total bill using the hist method on the Series:'''
tips = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_tips.csv')
tips.head()
fig, aix_hist = plt.subplots(2,1, sharex=True)

tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.head()

tips['tip_pct'].hist(bins=50, ax=aix_hist[0])

'''
A related plot type is a density plot, which is formed by computing an estimate 
of a continuous probability distribution that might have generated the observed 
data.
a usual procedure is to approcimate this distribution as a mixture of kernels, 
that is, simpler distributions like the normal (Gaussian) distribution. 
Thus, density plots are also known as KDE(kernel density estimate) plots.
Using plot with kind='kde' makes a density plot using the standard mixture-of-
normals KDE:'''
tips['tip_pct'].plot(kind='kde', ax=aix_hist[1])

plt.subplots_adjust(wspace=0, hspace=0)

'''
These two plot types are often plotted together; the histogram in normalized form
(to give a binned density) with a kernel density estimate plotted on top. 
As an example, consider a bimodal distribution consisting of draws from two 
different dtandard normal distributions:'''
comp1 = np.random.normal(0,1, size=200)  # N(0,1)
comp2 = np.random.normal(10,2, size=200)  # N(10,2)

values = pd.Series(np.concatenate([comp1,comp2]))
values

values.hist(bins=100, alpha=0.3, color='k', normed=True)

values.plot(kind='kde', style='k--')

values = pd.Series(np.concatenate([comp1,comp2]))


## 2.4 Scatter Plots
'''
Scatter plots are a useful way of examining the relationship between two 
one-dimensional data series. matplotlib has a scatter plotting method. 
I load the macrodata dataset from the statsmodels project, select a few variables,
then compute log differences:'''
macro = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                    '\macrodata.txt')
macro.head()

data = macro[['cpi','m1','tbilrate','unemp']]
data.head()
data.shape

# compute log differences:
trans_data = np.log(data).diff().dropna()
trans_data.tail()    

# plot a simple scatter plot using plt.scatter:
plt.scatter(trans_data['m1'], trans_data['unemp'])

plt.title('Changes in log %s vs. log %s' % ('m1', 'unemp'))

'''
In exploratory data analysis it's helpful to be able to look at all the scatter 
plots among a group of variables; this is known as a pairs plot or scatter plot 
matrix.
pandas has a scatter_matrix function for creating one from a DataFrame. It also 
supports placing histograms or density plots of each variable along the diagonal:'''
trans_data.head()

pd.scatter_matrix(trans_data, diagonal='kde', color='r', alpha=0.3)



# 3. Plotting Maps: Visualizing Haiti Earthquake Crisis Data

data = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch08_Haiti.csv')
data.info()
data.head()
data.shape

'''
Each row represents a report sent from someone's mobile phone indicating an emergency
or some other problem. Each has an associated timestamp and a location as latitude
and longitude:'''
data[['INCIDENT DATE','LATITUDE','LONGITUDE']][:10]

'''
the CATEGORY field contains a comma-separated list of codes indicating the type of
message:'''
data['CATEGORY'][:6]

'''
If you notice above in the data summary, some of the categories are missing, so 
we might want to drop these data points.
Additionally, calling describe shows that there are some aberrant locations:'''

data.describe()

'''
Cleaning the bad locations and removing the missing categories is now fairly simple:'''
data = data[(data.LATITUDE > 18) & (data.LATITUDE < 20) &
            (data.LONGITUDE > -75) & (data.LONGITUDE < -70) &
            data.CATEGORY.notnull()]

'''
Now we might want to do some analysis or visualization of this data by category,
but each category field may have multiple categories. Additionally, each category
is given as a code plus an English and possible also a French code name. 

First, I wrote these two functions to get a list of all categories and to split 
each category into a code and an English name:'''

def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]

def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))

def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split('|')[1]
    return code, names.strip()


# test
get_english('1. Urgences | Emergency ')
    
'''
Now, I make a dict mapping code to name because we'll use the codes for analysis.
We'll use this later when adorning plots (note the use of a generator expression
in lieu of a list comprehension):'''

all_cats = get_all_categories(data.CATEGORY)
all_cats

# Generator expression
english_mapping = dict(get_english(x) for x in all_cats)
english_mapping

english_mapping['2a']

english_mapping['6c']

'''
There are many ways to go about augmenting the data set to be able to easily 
select records by category. One way is to add indicator (or dummy) columns, one
for each category.
To do that, first extraxt the unique category codes and construct a DataFrame
of zeros having those as its columns and the same index as data:'''

def get_code(seq):
    return [x.split('.')[0] for x in seq if x]

all_codes = get_code(all_cats)
list(all_codes)
list(english_mapping.keys())

code_index = pd.Index(np.unique(all_codes))

dummy_frame = pd.DataFrame(np.zeros((len(data), len(code_index)))
            ,index=data.index, columns=code_index)
dummy_frame.head()

dummy_frame.iloc[:,:6].info()

for row, cat in zip(data.index, data.CATEGORY):
    codes = get_code(to_cat_list(cat))
    dummy_frame.loc[row, codes]=1

data = data.join(dummy_frame.add_prefix('category_'))

data.iloc[:,10:15].info()


















