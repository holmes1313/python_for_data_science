

################# 6.Data Loading, Storage, and File Formats ######################
'''
Input and output typically falls into a few main categories: 
    1.reading text files and other more efficient on-disk formats 
    2.loading data from databases 
    3.interacting with network source like web APIs'''
    
import pandas as pd
 
# 6.1 Reading and Writing Data in Text Format

outfile = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science'\
             '\ch6_ex1.csv','w')
outfile.write('a,b,c,d,message\n')
outfile.write('1,2,3,4,hello\n')
outfile.write('5,6,7,8,world\n')
outfile.write('9,10,11,12,foo')

outfile.close()

'''
since this is comma-delimited, we can use read_csv to read it into a DataFrame:'''

df = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science'\
                 '\ch6_ex1.csv')
df
type(df)
# pandas.core.frame.DataFrame

'''
We could also have used read_table and specifying the delimiter:'''
pd.read_table(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science'\
                 '\ch6_ex1.csv', sep=',')


'''
A file will not always have a header row.
You can allow pandas to assign default column names, or you can specify names
yourself:'''
outfile = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex2.csv', 'w')
outfile.write('1,2,3,4,hello\n')
outfile.write('5,6,7,8,world\n')
outfile.write('9,10,11,12,foo\n')
outfile.close()

pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex2.csv', header=None)

pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex2.csv', names=['a','b','c','d','message'])


'''
Suppose you wanted the message column to be the index of the returned DataFrame.'''
names = ['a','b','c','d','message']
pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex2.csv', names=names, index_col='message')


'''
To form a hierarchical index from multiple columns, just pass a list of column
numbers or names:'''
outfile = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
            '\ch06_csv_mindex.csv', 'w')
outfile.write('key1,key2,value1,value2\n')
outfile.write('one,a,1,2\n')
outfile.write('one,b,3,4\n')
outfile.write('one,c,5,6\n')
outfile.write('one,d,7,8\n')
outfile.write('two,a,9,10\n')
outfile.write('two,b,11,12\n')
outfile.write('two,c,13,14\n')
outfile.write('two,d,15,16\n')
outfile.close()

parsed = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                    '\ch06_csv_mindex.csv', index_col=['key1','key2'])
parsed

     
'''
In some cases, a table might not have a fixed delimiter, using whitespace or 
some other pattern to separate fields. 
In these cases, you can pass a regular expression '\s+' as a delimiter for read_table.'''
outfile = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex3.txt','w')
outfile.write('            A        B        C\n')
outfile.write('aaa -0.264438 -1.43534 -0.63423\n')
outfile.write('bbb -0.243348 -1.43534 -0.63423\n')     
outfile.write('ccc -0.243348 -1.43534 -0.63423\n')      
outfile.write('ddd -0.243348 -1.43534 -0.63423\n') 
outfile.close()

pd.read_table(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex3.txt', sep='\s+')
'''
Because there was one fewer column name than the number of data rows, read_table
infers that the first column should be the DataFrame's index in this special 
case'''


'''
You can skip the first, third, and fourth rows of a file with skiprows:'''

outfile = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science'\
               '\ch06_ex4.csv', 'w')
outfile.write('# hey!\n')
outfile.write('a,b,c,d,message\n')
outfile.write('#just wanted to make things more difficult for you\n')
outfile.write('#who reads CSV files with computers, anyway?\n')
outfile.write('1,2,3,4,hello\n')
outfile.write('5,6,7,8,world\n')
outfile.close()

pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex4.csv', skiprows=[0,2,3])


'''
Missing data is usually either not present (empty string) or marked by some
sentinel values, such as NA, -1.#IND, and NULL:'''
outfile = open(r'C:\Users\z.chen7\Downloads\Python'\
              '\pyhton_for_data_science\ch06_ex5.csv', 'w')
outfile.write('something,a,b,c,d,message\n')
outfile.write('one,1,2,3,4,NA\n')
outfile.write('two,5,6,,8,world\n')
outfile.write('three,9,10,11,NULL,foo\n')
outfile.close()

result = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex5.csv')
result

pd.isnull(result)

'''
The na_values option can take either a list or set of strings to consider missing
values:'''
pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex5.csv', na_values=[1])

'''
Different NA sentinels can be specified for each column in a dict:'''
sentinels = {'message':['foo','NA'], 'something':['two']}
pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex5.csv', na_values=sentinels)



## 6.1.1 Reading Text files in pieces
'''
When processing very large files or figuring out the right set of arguments
to correctly process a large file, you may only want to read in a small piece 
of a file or iterate through smaller chunks of the file.

If you want to only read out a small number of rows (avoiding reading the
entire file), specify that with nrows:'''
a = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex6.csv', nrows=5)

'''
To read out a file in pieces, specify a chunksize as a number of rows:'''
chunker = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
               '\ch06_ex6.csv', chunksize=1000)
chunker

'''
The TextParser object returned by read_csv allows you to iterate over the parts
of the file according to the chunksize. 
For example, we can iterate over ch06_ex6.csv, aggregating the value counts 
in the 'key' column:'''

tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)
tot



## 6.1.2 Writing Data Out to Text Format
'''
Data can also be exported to delimited format.'''
data = pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_ex5.csv')
data

'''
Using DataFrame's to_csv method, we can write the data out to a comma-seperated
file:'''
data.to_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_out1.csv')

pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_out1.csv')


data.to_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_out2.csv', sep='|', na_rep='NULL',index=False,header=False)

pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_out2.csv')


'''
Other delimiters can be used (writing to sys.stdout so it just prints the text
result; make sure to import sys):'''
import sys

data.to_csv(sys.stdout, sep='|')


'''
Missing values appear as empty strings in the output. You might want to denote
them by some other sentinel value:'''
data.to_csv(sys.stdout, na_rep='NULL')


'''
With no other options specified, both the row and column labels are written.
Both of these can be disabled:'''
data.to_csv(sys.stdout, index=False, header=False)

'''
You can also write only a subset of the columns, and in an order of your choosing:'''
data.to_csv(sys.stdout, index=False, columns=['a','b','c'], na_rep='NULL', sep='\t')


'''
Series also has a to_csv method:'''
import numpy as np
dates = pd.date_range('1/1/2000', periods=7)    # or pd.date_range('1/1/2000', '1/7/2000')
ts = pd.Series(np.arange(7), index=dates)
ts

ts.to_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_tseries.csv')

pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_tseries.csv', header=None, index_col=0)


'''
With a bit of wrangling (no header, first column as index), you can read a CSV
version of a Series with read_csv, but there is also a from_csv convenience method
that makes it a bit simpler:'''
pd.Series.from_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
                   '\ch06_tseries.csv', parse_dates=True)




## 6.1.3 Manually Working with Delimited Formats

'''
Most forms of tabular data can be loaded from disk using functions like 
pd.read_table. In some cases, however, some manual processing may be necessary.
It's not uncommon to receive a file with one or more malformed lines that trip
up read_table.'''

outfile = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
            '\ch06_ex7.csv', 'w')
outfile.write("'a','b','c'\n")
outfile.write("'1','2','3'\n")
outfile.write("'1','2,'3','4'\n")
outfile.close()

pd.read_csv(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
            '\ch06_ex7.csv')
# ParserError: Error tokenizing data. C error: Expected 3 fields in line 3, saw 4


'''
For any file with a single-character delimiter, you can use Python's built-in 
csv module. To use it, pass any open file or file-like object to csv-reader:'''
import csv

f = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
            '\ch06_ex7.csv', 'r')
reader = f.read()
type(reader)


f2 = open(r'C:\Users\z.chen7\Downloads\Python\pyhton_for_data_science' \
            '\ch06_ex7.csv')
reader2 = csv.reader(f2)
'''
Iterating through the reader like a file yields tuples of values in each like 
with any quote characters removed:'''
for line in reader2:
    print(line)

'''
From there, it's up to you to do the wrangling necessary to put the data in
the form that you need it.'''
lines = list(csv.reader(f2))
lines
header, values = lines[0], lines[1:]
data_dict = {h : v for h, v in zip(header,zip(*values))}
data_dict


## 6.1.4 JSON Data
'''
JSON (JavaScript Object Notation) has become one of the standard formats for 
sending data by HTTP request between web browsers and other applications.'''



## 6.1.5 XML and HTML: Web Scraping
'''
Many websites make data available in HTML tables for viewing in a browser, but
not downloadable as an easily machine-readable format like JSON, HTML, or XML.'''

from lxml.html import parse
from urllib import urlopen

