

## 1.Date and Time Data Types and Tools
'''
datetime stores both the date and time down to the microsecond.'''
from datetime import datetime

now = datetime.now()
now

now.year
now.month
now.date()
now.time()
now.day

''' 
datetime.timedelta represents the temporal difference between two datetime objects:'''
delta = datetime(2018,4,13) - datetime.now()
delta

delta.days
delta.seconds

'''
You can add (or subtract) a timedelta or multiple thereof to a datetime object 
to yield a new shifted object:'''
from datetime import timedelta

start = datetime.now()

start + timedelta(12)

start - timedelta(2) * 2


### 1.1 Converting between string and datetime
'''
datetime objects and pandas Timestamp objects can be formatted as strings 
using str or the strftime method, passing a format specification:'''
stamp = datetime(2011,1,3)

str(stamp)

stamp.strftime('%m/%d/%Y')   # %m 2-digit month   %M 2-digit minute

''' 
These same format codes can be used to convert strings to dates using datetime.strptime:'''
value = '2001-0103'

datetime.strptime(value, '%Y-%m%d')

datestrs = ['7/6/2011','8/6/2011']

[datetime.strptime(date, '%m/%d/%Y') for date in datestrs]

'''
datetime.strptime is the best way to parse a date with a known format. 

However, it can be a bit annoying to have to write a format spec each time, 
especially for common date formats. 
In this case, you can use the parser.parse method in the third party
dateutil package:'''
from dateutil.parser import parse

parse('1991.04.13')

'''
dateutil is capable of parsing almost any human-intelligible date representation:'''
parse('apr 13, 1991, 8:15 pm')

'''
In international locales, day appearing before month is very common, so you can pass
dayfirst=True to indicate this:'''
parse('1/4/1991', dayfirst=True)

'''
pandas is generally oriented toward working with arrays of dates, whether used 
as an axis index or a column in a DataFrame. 
The to_datetime method parses many different kinds of date representations. 
Standard date formats like ISO8601 can be parsed very quickly.'''
import pandas as pd 

datestrs = ['7/12/2012 12:02', '4/13/1991']

pd.to_datetime(datestrs)

'''
It also handles values that should be considered missing (None, empty string, etc.):'''
idx = pd.to_datetime(datestrs + [np.nan])
idx

idx[-1]

idx.notnull()
idx.isnull()

'''
NaT (Not a Time) is pandas's NA value for timestamp data.'''


## 2. Time Series Basics
'''
The most basic kind of time series object in pandas is a Series indexed by 
timestamps, which often represented external to pandas as Python strings or
datetime objects:'''
dates = [datetime(2011,1,2), datetime(2011,1,5), datetime(2011,1,7),
         datetime(2011,1,8), datetime(2011,1,10), datetime(2011,1,12)]

ts = pd.Series(np.random.randn(6), index=dates)
ts

'''
Under the hood, these datetime objects have been put in a DatetimeIndex, and the 
the variable ts is now type TimeSeries:'''
type(ts.index)

ts.index

'''
Like other Series, arithmetic operations between differently-indexed time series automatically
align on the dates:'''
ts[::2]
ts

ts + ts[::2]

'''
Scalar values from a DatetimeIndex are pandas Timestamp objects'''
ts

stamp = ts.index[0]
stamp

ts[stamp]
ts['20110102']
ts[datetime(2011,1,2)]
ts['01/02/2011']


### 2.1 Indexing, Selection, Subsetting
dates = [datetime(2011,1,2), datetime(2011,1,5), datetime(2011,1,7),
         datetime(2011,1,8), datetime(2011,1,10), datetime(2011,1,12)]
ts = pd.Series(np.random.randn(6), index=dates)
ts

ts[ts.index[2]]

ts['1/10/2011']

ts['20110110']

ts['2011.01.02']

'''
For longer time series, a year or only a year and month can be passed to easily 
select slice of data'''
longer_ts = pd.Series(np.random.randn(1000),
                      index = pd.date_range('1/1/2000',periods=1000))
y = lambda g: g.year
longer_ts.groupby(y).apply(len)

len(longer_ts['2001'])
longer_ts['2001']

longer_ts['2001-05']

'''
Slicing with dates works just like with a regular Series:'''
dates = [datetime(2011,1,2), datetime(2011,1,5), datetime(2011,1,7),
         datetime(2011,1,8), datetime(2011,1,10), datetime(2011,1,12)]
ts = pd.Series(np.random.randn(6), index=dates)
ts

ts['20110107':'20110110']

'''
Because most time series data is ordered chronologically, you can slice with 
timestamps not contained in a time series to perform a range query:'''
ts

ts['1/6/2011':'1-11-2011']


'''
There is an equivalent instance method truncate which slices a TimeSeries between
two dates:'''
ts
ts.truncate(before='20110106', after='20110112')

'''
All of the above holds true for DataFrame as well, indexing on its rows:'''
dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100,4),
                       index=dates,
                       columns=['Colorado','Texas','New York','Ohio'])
long_df

long_df.loc['5-2001']
long_df.loc['2001-05']


### 2.2 Time Series with Duplicate Indices
'''
In some applications, there may be multiple data obervations falling on a particular 
timestamp:'''
dates = pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/2/2000','1/3/2000'])

dup_ts = pd.Series(np.arange(5),index=dates)
dup_ts

'''
We can tell that the index is not unique by checking its is_unique property:'''
dup_ts.index.is_unique

dup_ts['2000-01-02']

'''
Suppose you wanted to aggregate the data having non-unique timestamps. One way
to do this is to use groupby and pass level=0 (the only level of indexing!):'''
grouped = dup_ts.groupby(level=0)

grouped.sum()
grouped.mean()
grouped.count()


## 3. Date Rangers, Frequencies, and Shifting
'''
In the example time series, converting it to be fixed daily frequency can be 
accomplished by calling resample:'''
dates = [datetime(2011,1,2), datetime(2011,1,5), datetime(2011,1,7),
         datetime(2011,1,8), datetime(2011,1,10), datetime(2011,1,12)]
ts = pd.Series(np.random.randn(6), index=dates)
ts

pd.Series(ts.resample('D')).fillna(0)

### 3.1 Generating Date Rangers
'''
pd.date_range is responsible for generating a DatetimeIndex with an indicated
length according to a particular frequency:'''
len(pd.date_range('4/1/2011', '6/1/2012'))  # both ends are included

'''
By default, date_range generates daily timestamps. If you pass only a start
or end date, you must pass a number of periods to generate:'''
pd.date_range(start='4.1.2011', periods=11)

pd.date_range(end='4/11/2011', periods=11)

'''
The start and end dates define strict boundaries for the generated date index. 
For example, if you wanted a date index containing the last business day of each
month, you would pass the 'BM' frequency (business end of month) and only dates
falling on or inside the date interval will be included:'''
pd.date_range('1/1/2018','12/31/2018',freq='BM')  # business end of month

'''
date_range by default preserves the time (if any) of the start or end timestamp:'''
pd.date_range('5/2/2012 12:56:31', periods=4)

'''
If you want to generate a set of timestamps normalized to midnight as a convention,
there is a normalize option:'''
pd.date_range('5.2.2012 12:56:31',periods=4, normalize=True)


### 3.2 Frequencies and Date Offsets
'''
For each base frequency, there is an object defined generally referred to as 
a date offset.
For example, hourly frequency can be represented with the Hour class:'''
from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour

four_hours = Hour(4)
four_hours

'''
Many offsets can be combined together by condition:'''
Hour(2) + Minute(30)


'''
In most applications, you would use a string like 'H' or '4H'. Puttting an 
integer before the base frequency creates a multiple:'''
pd.date_range('1/1/2018','1/3/2018',freq='4H')

pd.date_range('1/1/2018','1/3/2018',freq='1H30min')

'''
Some frequencies describe points in time that are not evenly spaced, such as 
'M' (calendar month end) and 'BM' (last business/weekday of month).
For lack of a a better term, I call these anchored offsets.'''

## Week of month dates
'''
One useful frequency class is 'week of month', starting with WOM. 
This eables you to get dates like the third Friday of each month:'''
rng = pd.date_range('1/1/2018','9/1/2018',freq='WOM-3FRI')
list(rng)


### 3.3 Shifting (Leading and Lagging) Data
'''
"shifting" refers to moving data backward and forward through time. Both Series
and DataFrame have a shift method for doing naive shifts forward and backward, 
leaving the index unimodified:'''
ts = pd.Series(np.random.randn(4),
               index = pd.date_range('1/1/2000',periods=4, freq='M'))
ts

ts.shift(2)
ts.shift(-2)

'''
a common use of shift is computing percent changes in a time series or multiple 
time series as DataFrame columns:'''
ts
ts.shift(1)

ts / ts.shift(1) -1

ts.pct_change()

'''
Because naive shifts leave the index unmodified, some data is discarded. Thus
if the frequency is known, it can be passed to shift to advance the timestamps
instead of simply the data:'''
ts

ts.shift(2,freq='M')

ts.shift(-2, freq='D')
ts.shift(-1, freq='2D')

ts.shift(1,freq='90T')  # minutes

# shifting dates with offsets
from pandas.tseries.offsets import Day, MonthEnd

now = datetime.now()
now
now + 3*Day()

'''
If you add an anchored offset like MonthEnd, the first increment will roll 
forward a date to the next date according to the frequency rule:'''
now + MonthEnd()

now + MonthEnd(2)

'''
Anchored offsets can explicitly "roll" dates forward or backward using their
rollforward and rollback methods, respectively:'''
offset = MonthEnd()

offset.rollforward(now)

offset.rollback(now)

'''
A clever use of date offset is to use these methods with groupby:'''
ts = pd.Series(np.random.randn(20),
               index=pd.date_range('1/15/2000', periods=20, freq='4d'))
ts

ME = lambda g: offset.rollforward(g)
ts.groupby(ME).apply(len)
ts.groupby(ME).mean()

# pay attention to the difference
ts.shift(1,freq='M')
ts.shift(1,freq='M').groupby(level=0).apply(len)
ts.shift(1,freq='M').groupby(level=0).mean()

'''
Of course, an easier and faster way to do this is using resample (much more 
on this later):'''
ts
ts.resample('M')

pd.Series(ts.resample('D'))

ts.resample('M').mean()

ts.resample('MS').mean()


## 4. Time Zone Handling
'''
Time zones are expressed as offsets from UTC (coordinated universal time).
In Python, time zone information comes from the 3rd party pytz library.'''
import pytz

'''
Time zone names can be found interactively and in the docs:'''
pytz.common_timezones[-5:]

'''
To get a time zone object from pytz, use pytz.timezone:'''
tz = pytz.timezone('US/Eastern')
tz


### 4.1 Localization and Conversion
'''
By default, time series in pandas are time zone naive.'''
rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts

'''
The index's tz field is None:'''
print(ts.index.tz)

'''
Date ranges can be generated with a time zone set:'''
pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')

'''
Conversion from naive to localized is handled by the tz_localized method:'''
ts_utc = ts.tz_localize('UTC')
ts_utc

ts_utc.index

'''
Once a time series has been localized to a particular time zone, it can be 
converted to another time zone using tz.convert:'''
ts_utc.tz_convert('US/Eastern')

'''
In the case of the above time series, which straddles a DST transition in the 
US/Eastern time zone, we would localize to EST and convert to, say, UTC or Berlin
time:'''
ts_eastern = ts.tz_localize('US/Eastern')

ts_eastern.tz_convert('UTC')

ts_eastern.tz_convert('Europe/Berlin')

'''
tz_localize and tz_convert are also instance methods on DatetimeIndex:'''
ts.index.tz_localize('Asia/Shanghai')


### 4.2 Operations with Time Zone-aware Timestamp Objects
'''
Individual Timestamp objects similarly can be localized from naive to time zone-aware
and convert from one time zone to another:'''
stamp = pd.Timestamp('2011-03-12 4:00')

stamp_utc = stamp.tz_localize('utc')

stamp_utc.tz_convert('US/Eastern')

'''
You can also pass a time zone when creating the Timestamp:'''
stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
stamp_moscow

'''
When performing time arithmetic using the pandas's DateOffset objects, daylight
savings time transitions are respected where possible:'''
# 30 mins before DST transition
from pandas.tseries.offsets import Hour
stamp = pd.Timestamp('2012-03-12 01:30',tz='US/Eastern')
stamp
stamp + Hour()

# 90 mins before DST transition
stamp = pd.Timestamp('2012-11-04 00:30', tz = 'US/Eastern')
stamp
stamp + 2*Hour()


### 4.3 Operations between Different Time Zones
'''
If two time series with different time zones are combined, the result will be
UTC. Since the timestamps are stored under the hood in UTC, this is a straightforward
operation and requires no conversion to happen:'''
rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
rng
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts

ts1 = ts[:7].tz_localize('Europe/London')

ts2 = ts1[2:].tz_convert('Europe/Moscow')

result = ts1 + ts2
result
result.index


## 5. Periods and Period Arithmetic
'''
Periods represent time spans, like days, months, quarters, or years. The Period
class represents this data type, requiring a string or interger and a frequency
from the above table:'''
p = pd.Period(2007, freq='A-DEC')
p

'''
In this case, the Period object represents the full timespan from Jan 1, 2007
to Dec 31, 2007, inclusive.
Conveniently, adding and substracting integers from periods has the effect of
shifting by their frequency:'''
p+5
p-2

'''
If two periods have the same frequency, their difference is the number of units
between them:'''
pd.Period('2014', freq='A-DEC') - p

'''
Regular ranges of periods can be constructed using the period_range function:'''
rng = pd.period_range('1/1/2000','6/30/2000',freq='M')
rng

'''
The PeriodIndex class stores a sequence of periods and can serve as an axis index
in any pandas data structure:'''
pd.Series(np.random.randn(len(rng)), index=rng)

'''
If you have an array of strings, you can also appeal to the PeriodIndex class
itself:'''
values = ['2001Q3','2002Q2','2003Q1']

index = pd.PeriodIndex(values, freq='Q-DEC')
index 
 

### 5.1 Period Frequency Conversion
'''
Periods and PeriodIndex objects can be converted to another frequency using 
their asfreq method.
As an example, suppose we had an annual period and wanted to convert it into 
a monthly period either at the start or end of the year:'''
p = pd.Period('2007', freq='A-DEC')
p

p.asfreq('M')
p.asfreq('M', how='end')
p.asfreq('M', how='start')

'''
You can think of Period('2007','A-DEC') as being a cursor pointing to a span 
of time, subdivided by monthly periods.
For a fiscal year ending on a month other than December, the monthly 
subperiods belonging are different:'''
p = pd.Period(2007, freq='A-JUN')
p

p.asfreq('M', how='end')   #Period('2007-06', 'M')
p.asfreq('D', how='start')   #Period('2006-07-01', 'D')

'''
When converting from high to low frequency, the superperiod will be determined 
depending on where the subperiod "belongs".
For example, in A-JUN frequency, the month Aug-2007 is actually part of the 
2008 period:'''
p = pd.Period('8/2007', 'M')
p

p.asfreq('A-JUN')
# Period('2008', 'A-JUN'): JUL 2007 - Jun 2008

pd.Period('7/2007').asfreq('A-JUN')

p.asfreq('A-DEC')

'''
Whole Period objects or TimeSeries can be similarly converted with the same
semantics:'''
rng = pd.period_range(2006, 2009, freq='A-DEC')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts

ts.asfreq('M', how='start')

ts.asfreq('B', how='end')
ts.asfreq('D', how='end')


### 5.2 Quarterly Period Frequencies
'''
Much quarterly data is reported relative to a fiscal year end, typically 
the last calendar or business day of one of the 12 months of the year. As 
such, the period 2012Q4 has a different meaning depending on fiscal year end.
pandas supports all 12 possible quarterly frequencies as Q-JAN through Q-DEC:'''
p = pd.Period('2012Q4', freq='Q-JAN')  # 2011.11 - 2012.1

p.asfreq('M', how='start')
p.asfreq('M', how='end')

'''
In the case of fiscal year ending in January, 2012Q4 runs from November through
January, which you can check by converting to daily frequency:'''
p.asfreq('D','start')
p.asfreq('D','end')

'''
Thus, it's possible to do period arithmetic very easily; for example, to get
the timestamp at 4PM on the 2nd to last business day of the quarter:'''
p = pd.Period('2018Q4', freq='Q-DEC')

# the last business day of the quarter
lb = p.asfreq('B', how='end')

# the 2nd to last business day of the quarter
lb_2 = lb - 1

# the 00:00 of the 2nd to last business day 
beg_lb_2 = lb_2.asfreq('T', how='start')

# 4PM on the 2nd to last business day of the quarter
p4pm  = beg_lb_2 + 16*60
p4pm

# or
(p.asfreq('B', how='end') - 1).asfreq('T', how='start') + 16*60

p4pm.to_timestamp()

p4pm.to_timestamp()

'''
Generating quarterly ranges works as you would expect using period_range.
Arithmetic is identical, too:'''
rng = pd.period_range('2018q3','2019q4', freq='Q-DEC')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts

new_rng = (ts.index.asfreq('B',how='end')-1).asfreq('T',how='start') + 16*60
new_rng

ts.index = new_rng
ts
type(ts.index)


### 5.3 Converting Timestamps to Periods (and Back)
'''
Series and DataFrame objects indexed by timestamps can be converted to periods
using the to_period method:'''
rng = pd.date_range('1/1/2000',periods=3, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts

pts = ts.to_period()
pts
type(pts.index)

pts.index = pts.index.asfreq('D', how='end').to_timestamp()
pts
ts

'''
Since periods always refer to non-overlapping timespans, a timestamp can only 
belong to a single period for a given frequency.
While the frequency of the new PeriodIndex is inferred from the timestamps by 
default, you can specify any frequency you want. There is also no problem with
having fuplicate periods in the result:'''
rng = pd.date_range('1/29/2000', periods=6)
ts2 = pd.Series(np.arange(len(rng)), index=rng)
ts2

ts2.to_period()
ts2.to_period('M')
ts2.to_period('Q-DEC')
ts2.to_period('A-DEC').to_period()

'''
To convert back to timestamps, use to_timstamp:'''
rng = pd.date_range('1/2000', periods=6, freq='M')
ts3 = pd.Series(np.arange(len(rng)), index=rng)
ts3

pts = ts3.to_period('M')
pts

pts.to_timestamp(how='end')

### 5.4 Creating a PeriodIndex from Arrays
'''
Fixed frequency data sets are sometimes stored with timespan information spread
across multiple columns.
For example, in this macroeconomic data set, the year and quarter are in different
columns:'''
data = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\macrodata.txt')
data.head()

data.year
data.quarter

'''
By passing these arrays to PeriodIndex with a frequency, they can be combined
to form an index for the DataFrame:'''
index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
index

data.index= index
data.head()

data.asfreq('D', how='end').head()



## 6. Resampling and Frequncy Conversion
'''
Resampling refers to the process of converting a time series from one freqency
to another.
Aggregating higher frequency data to lower frequency is called downsampling,
while converting lower frequency to higher frequency is called upsampling.
Not all resampling falls into either of these categories; for example,
converting W-WED (weekly on Wednesday) to W-FRI is neither upsampling nor 
downsampling.
pandas objects are equipped with a resample method, which is the workhorse
function for all frequency conversion:'''
rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts[:10]

ts.resample('M').mean()
# or
ts.groupby(lambda g: g.month).mean()

ts.resample('M',kind='period').mean()

ts.resample('M').mean().to_period('M')


### 6.1 Downsampling
'''
The data you're aggregating doesn't need to be fixed frequently; the desired 
frequency defines bin edges that are used to slice the time series into pieces 
to aggregate.
Each interval is said to be half-open; a data point can only belong to one 
interval, and the union of the intervals must make up the whole time frame.
When using resample to downsample data, you need to consider:
    1.Which side of each interval is closed
    2.How to label each aggregated bin, either with the start of the interval or
the end
'''
rng = pd.date_range('1/1/2000',periods=12, freq='T')
ts = pd.Series(np.arange(len(rng)), index=rng)
ts

'''
Suppose you wanted to aggregate this data into five-minute chunk or bars by 
taking the sum of each group:'''
ts.resample('5min').sum()

'''
By default, the left bin edge is inclusive, so the 00:00 value is included in
the 00:00 to 00:05 interval. 
Passing closed='right' changes the interval to be closed on the right:'''
ts.resample('5min', closed='right').sum()

'''
As you can see, the resulting time series is labeled by the timestamp from the
left side of each bin.
By passing label='right' you can label them with the right bin edge:'''
ts.resample('5min',closed='right', label='right').sum()

'''
Lastly, you might want to shift the result index by some amount, say subtracting
one second from the right edge to make it more clear which interval the timestamp
refers to. 
To do this, pass a string or date offset to loffset:'''
ts.resample('5min', closed='right', label='right', loffset='-1s').sum()

# Open-High-Low-Close (OHLC) resampling
'''
In finance, an ubiquitous way to aggregate a time series is to compute four values
for each bucket: the first (open), last (close), maximum (high), and minimal (low)
values.
By passing how='ohlc''''
ts.resample('5min', how='ohlc')


# Resampling with GroupBy
'''
An alternative way to downsample is to use pandas's groupby functionality.
For example, you can group by month or weekday by passing a function that 
accesses those fields on the time series's index:'''
rng = pd.date_range('1/1/2000',periods=100, freq='D')
ts = pd.Series(np.arange(100), index=rng)
ts[:10]

ts.resample('M').mean()
# or
ts.groupby(lambda g: g.month).mean()

ts.groupby(lambda g: g.weekday).mean()


### 6.2 Upsampling and Interplolation
'''
Let's consider a DataFrame with some weekly data:'''
frame = pd.DataFrame(np.random.randn(2,4),
                     index = pd.date_range('1/1/2000',periods=2, freq='W-WED'),
                     columns = ['Colorado','Texas','New York','Ohio'])
frame

'''
When resampling this to daily frequency, by default missing value are 
introduced:'''
df_daily = frame.resample('D')
df_daily

'''
Suppose you wanted to fill forward each weekly value on the non-Wednesday. 
The same filling or interplolation methods available in the fillna and reindex
methods are available for resampling:'''
frame.resample('D', fill_method='ffill')

frame.resample('D', fill_method='bfill')

'''
You may similarly choose to only fill a certain number of periods forward to limit
how far to continue using an observed value:'''
frame.resample('D', fill_method='ffill', limit=1)

'''
Notably, the new data index need not overlap with the old one at all:'''
frame.resample('W-THU', fill_method='ffill')

frame.resample('W-THU', fill_method='bfill')


### 6.3 Resampling with Periods
frame = pd.DataFrame(np.random.randn(24,4),
                     index=pd.period_range('1-2000','12-2001',freq='M'),
                     columns=['Colorado','Texas','New York','Ohio'])
frame

annual_frame = frame.resample('A-DEC').mean()
annual_frame

frame.asfreq('A-DEC').groupby(level=0).mean()

# Q-DEC: Quarterly, year ending in December
annual_frame.resample('Q-DEC', fill_method='ffill')

annual_frame.resample('Q-DEC', fill_method='ffill').to_timestamp()



## 7. Time Sries Plotting
close_px_all = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                           '\ch09_stock_px.csv', parse_dates=True, index_col=0)
close_px_all.head()

close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px[-10:]

close_px1 = close_px.resample('B').ffill()

close_px.info()
close_px1.info()

'''
Calling plot on one of the columns gernerates a simple plot:'''
close_px1['AAPL'].plot()

'''
When called on a DataFrame, all of the time series are drawn on a single subplot
with a legend indicating which is which. 
I'll plot only the year 2009 data so you can see how both months and years are
formatted on the X axis:'''
close_px1.loc['2009', 'AAPL'].plot()

close_px1.loc['2009'].plot()

close_px1['AAPL'].loc['2011-01':'2011-03'].plot()

'''
Quarterly frequency data is also more nicely formatted with quarterly markers, 
something that would be quite a bit more work to do by hand:'''
close_px1[:10]

appl_q = close_px1['AAPL'].resample('Q-DEC').ffill()



## 8. Moving Windonw Functions
'''
rolling_mean takes a TimeSeries or DataFrame along with a window (expressed
as a number of periods):'''
close_px_all = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                           '\ch09_stock_px.csv', parse_dates=True, index_col=0)
close_px_all.head()

close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px[-10:]

close_px = close_px.resample('B').ffill()

close_px.AAPL.plot()

pd.rolling_mean(close_px.AAPL, 250).plot()

pd.rolling_mean(close_px.AAPL, 100).plot()
                                                               
                                                                  
'''
By default functions like rolling_mean require the indicated number of non-NA
observations. This behavior can be changed to account for missing data and, in
particular, the fact that you will have fewer than window periods of data at the
beginning of the time series:'''

appl_std250 = pd.rolling_std(close_px.AAPL, 250, min_periods=10)

appl_std250[5:12]                                                                  

appl_std250.plot()                                                                


'''
To compute an expanding window mean, you can see that an expanding window is
just a special case where the window is the length of the time series, but 
only one or more periods is required to compute a value:'''

# Define expanding mean in terms of rolling_mean
expanding_mean = lambda x: rolling_mean(x, len(x), min_periods=1)

'''
Calling rolling_mean and friends on a DataFrame applies the transformation to
each column:
'''
pd.rolling_mean(close_px, 60).plot(logy=True)


### 8.1 Exponentially-weighted functions
'''
An alternative to using a static window size with equally-weighted observations 
is to specify a constant decay facor (span) to give more weight to more recent 
observations.
Since an exponentially-weighted statistic places more weight on more recent 
observations, it 'adapts' faster to changes compared with the equal-weighted
version.
Here's an example comparing a 60-day moving average of Apple's stock price
with an EW moving average with span=60:'''
import matplotlib.pyplot as plt
%matplotlib

fig, axes = plt.subplots(2,1,sharex=True,sharey=True, figsize=(12,7))

close_px_all = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                           '\ch09_stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px = close_px.resample('B').ffill()

aapl_px = close_px.AAPL['2005':'2009']

ma60 = pd.rolling_mean(aapl_px, 60, min_periods=50)

ewma60 = pd.ewma(aapl_px, span=60)
 
axes[0].plot(aapl_px, 'k-')
axes[0].plot(ma60, 'k--')

axes[1].plot(aapl_px, 'k-')
axes[1].plot(ewma60, 'r--')


### 8.2 Binary Moving Window Functions
'''
Some statistical operators, like correlaiton and covariance, need to operate
on two time series. As an example, financial analysts are often interested
in a stock's correlation to a benchmark index like the S&P 500.
We can compute that by computing the percent changes and using 
rolling_corr:'''
close_px_all = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                           '\ch09_stock_px.csv', parse_dates=True, index_col=0)
close_px_all[:10]
spx_px = close_px_all['SPX']
spx_px = spx_px.resample('B').ffill()
spx_px[:10]

spx_rets = spx_px.pct_change().dropna()

spx_rets1 = (spx_px / spx_px.shift(1) -1).dropna()

returns = close_px.pct_change().dropna()

corr = pd.rolling_corr(returns.AAPL, spx_rets, 125, min_periods=100)
corr.plot()

corr_all = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
corr_all.plot()


### 8.3 User-Defined Moving Window Functions
'''
The rolling_apply function provides a means to apply an array function of
 your own devising over a moving window. The only requirement is that the 
 function produce a single value (a reduction) fro each piece of the array.
 For example, while we can compute sample quantiles using rolling_quantile, we
 might be interested in the percentile rank of a particular value over the 
 sample. The scipy.stats.percentileofscore function does just this:'''
from scipy.stats import percentileofscore

score_at_2percent = lambda x: percentileofscore(x, 0.02)

result = pd.rolling_apply(returns.AAPL, 250, score_at_2percent)
result.plot()


## 9. Performance and Memory Usage Notes








