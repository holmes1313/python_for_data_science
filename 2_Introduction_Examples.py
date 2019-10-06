# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:18:46 2018

@author: z.chen7
"""

## Chapter 2: Introductory Examples

# 1.usa.gov data from bit.ly

path = r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
        r'\pydata-book-1st-edition\ch02\usagov_bitly_data2012-03-16-1331923249.txt'
open(path).readline()

import json

records = [json.loads(line) for line in open(path, 'rb')]
records

records[0]
records[0]['tz']


# Counting Time Zones in Pure Python
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
time_zones[:10]

def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

counts = get_counts(time_zones)

counts['America/New_York']
counts['']
len(time_zones)
len(counts)

def top_counts(count_dict, n):
    key_value_desc = [(k, count_dict[k]) for k in sorted(count_dict, key=count_dict.get, reverse=True)]
    return key_value_desc[:n]
top_counts(counts,10)
# *sort a dicitonary by it's value
[(k, counts[k]) for k in sorted(counts, key=counts.get, reverse=True)]

def top_counts2(count_dict, n):
    key_value_pairs = [(count, tz) for tz, count in count_dict.items()]
    key_value_pairs.sort()
    return key_value_pairs[-n:]
top_counts2(counts,10)



from collections import Counter

counts = Counter(time_zones)
counts.most_common(10)


# Counting Time Zones with pandas
type(records)
type(records[0])

import pandas as pd

frame = pd.DataFrame(records)
frame.shape
frame.info()

frame['tz'][:10]

tz_counts = frame['tz'].value_counts()
tz_counts[:10]

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz==''] = 'Unknown'

# diff between '' and NaN
frame['tz'][frame['tz'].isnull()]
frame['tz'][frame['tz']=='']
frame['tz'][:15]

tz_counts = clean_tz.value_counts()
tz_counts[:10]
tz_counts['Unknown']
tz_counts['Missing']

tz_counts[:10].plot(kind='barh', rot=0)


frame['a'][1]
frame['a'][50]
frame['a'][51]

len(frame['a'])
len(frame['a'].dropna())

results = pd.Series([x.split()[0] for x in frame['a'].dropna()])
results[:5]
results.value_counts()[:8]

cframe = frame[frame['a'].notnull()]
sum(cframe['a'].isnull())
len(cframe)

import numpy as np

operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
operating_system[:5]

by_tz_os = cframe.groupby(['tz',operating_system])
by_tz_os

by_tz_os.size()
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

agg_counts.sum(axis=1)
indexer = agg_counts.sum(axis=1).argsort()
indexer[:10]

count_subset = agg_counts.take(indexer)[-10:]
count_subset

count_subset.plot(kind='barh', stacked=True)

normed_subset = count_subset.div(count_subset.sum(axis=1), axis=0)
normed_subset.plot(kind='barh', stacked=True)


# MovieLens 1M Data set
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                      r'\pydata-book-1st-edition\ch02\movielens\users.dat',
                      sep='::', header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                        r'\pydata-book-1st-edition\ch02\movielens\ratings.dat',
                        sep='::', header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                       r'\pydata-book-1st-edition\ch02\movielens\movies.dat',
                       sep='::', header=None, names=mnames, engine='python')

data = pd.merge(pd.merge(ratings, users), movies)
data.head()

data.iloc[0]

mean_ratings = data.pivot_table(values='rating', index='title', columns='gender', aggfunc='mean')

ratings_by_title = data.groupby(by='title').size()
type(ratings_by_title)   # Series

active_titles = ratings_by_title.index[ratings_by_title >= 250]
active_titles

mean_ratings.head()
mean_ratings = mean_ratings.ix[active_titles]
mean_ratings.head()

top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
top_female_ratings[:10]

# Measuring rating disagreement
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
mean_ratings.head()

sort_by_diff = mean_ratings.sort_values(by='diff')
sort_by_diff[:15]

# Reverse order of rows, take first 15 rows
sort_by_diff[::-1][:15]


# Standard deviation of rating grouped by title
rating_std_by_title = data.groupby(by='title')['rating'].std()

# Filter down to active titles
ratings_by_title = data.groupby(by='title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
rating_std_by_title = rating_std_by_title.loc[active_titles]

# Order Series by value in descending order
rating_std_by_title.sort_values(ascending=False)[:10]


# US Baby Names 1880-2010
import pandas as pd

names1880 = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python' \
                        r'\pyhton_for_data_science\pydata-book-1st-edition\ch02' \
                        r'\names\yob1880.txt', names=['name','sex','births'])
names1880.head()

names1880.groupby('sex')['births'].sum()   # Series
names1880.groupby('sex').agg({'births':sum})   #DataFrame

# 2010 is the last available year right now
years = range(1880, 2011)

pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
           r'\pydata-book-1st-edition\ch02\names\yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

pieces[0].head()

# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)
'''
pd.concat glues the DataFrame objects together row-wise by default.'''

# aggregate the data at the year and sex level using groupby or pivot_table
names.shape
names.head()

names.groupby(['year', 'sex']).agg({'births':sum}).unstack().tail()

total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
total_births.tail()
total_births.plot(title='Total births by sex and year')


# insert a col prop with the fraction of babies given each name relative to the 
# total number of births
def add_prop(group):
    # Integer division floors
    births = group['births'].astype(float)
    group['prop'] = births / births.sum()
    return group

names = names.groupby(['year','sex']).apply(add_prop)
names
names.loc[(names['year']==1880) & (names['sex']=='F'), 'prop'].sum()

# the top 1000 names for each sex/year combination
pieces = []

for year, group in names.groupby(['year','sex']):
    top_1000_group = group.sort_values(by='births', ascending=False)[:1000]
    pieces.append(top_1000_group)
    
top1000 = pd.concat(pieces, ignore_index=True)
top1000    
    
    
# Analyzing Naming Trends
top1000.head()

boys = top1000[top1000['sex']=='M']
girls = top1000[top1000['sex']=='F']    

total_births = top1000.pivot_table('births', index='year', columns='name',
                                   aggfunc=sum)
total_births.info()

subset = total_births[['John','Harry','Mary','Marilyn']]
subset

subset.plot(subplots=True, figsize=(12,10), grid=False,
            title='Number of birth per year')


# Measuring the increase in naming diversity
top1000.head()
table = top1000.pivot_table('prop', index='year', columns='sex',
                            aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',
           yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
'''
From the plot, there appears to be increasing name diversity (decreasing
total proportion in the top 1000).'''

                                                            
'''
the number of distinct names, taken in order of popularity from highest to lowest,
in the top 50% of births.'''
df = boys[boys['year'] == 2010]
df
prop_cumsum = df.sort_values(by='prop', ascending=False)['prop'].cumsum()
prop_cumsum[:10]
prop_cumsum.values.searchsorted(0.5) + 1

df = boys[boys['year'] == 1900]
in1900 = df.sort_values(by='prop', ascending=False)['prop'].cumsum()
in1900.values.searchsorted(0.5) + 1

def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)['prop'].cumsum()
    return group.values.searchsorted(q)+1

diversity = top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity
diversity[(1900,'M')]
diversity = diversity.unstack('sex')
diversity.head()

diversity.plot(title='Number of popular names in top 50%')


# The "Last letter" Revolution
# extract last letter from name column
get_last_letter = lambda x: x[-1]
names.head()
last_letters = names['name'].map(get_last_letter)
last_letters.name = 'last_letter'
last_letters

table = names.pivot_table('births', index=last_letters, columns=['sex','year'],
                          aggfunc=sum)
table

subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
subtable.head()

# normalize the table
subtable.sum()

letter_prop = subtable / subtable.sum().astype(float)
letter_prop
letter_prop['F'][1910].sum()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,1,figsize=(10,8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female',
           legend=False)

# select a subset of letters for the boy names
letter_prop = table / table.sum().astype(float)
letter_prop

dny_ts = letter_prop.loc[['d','n','y'], 'M'].T
dny_ts.head()

dny_ts.plot()


# Boy names that became girl names (and versa)
all_names = top1000['name'].unique()

mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like

filtered = top1000[top1000['name'].isin(lesley_like)]
filtered
filtered.groupby('name')['births'].sum()

table = filtered.pivot_table('births', index='year', columns='sex', aggfunc=sum)
table = table.div(table.sum(1), axis=0)
table.tail()
table.plot(style={'M':'k-','F':'k--'})


x = 5
y = 7
if x>5:
    x += 1
    
    y = 8
















