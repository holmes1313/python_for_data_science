
# Dynamic references, strong types
a = 4.5
b = 2

print("a is %s, b is %s" % (type(a), type(b)))



# Binary operators and comparison
'''
To check if two references refer to the same object, use the is keyword. 
is not is also perfectly valid if you want to check that two objects are 
not the same:'''

a = [1,2,3]
b = a

'''Note, the list function always creates a new list'''
c = list(a)

a is b

a is not c

'''
Note this is not the same thing is comparing with =='''
a == c

'''
A very common use of is and is not is to check if a variable is None, since there is only
one instance of None:'''
a = None

a is None

''' Floor-divide a by b, dropping any fractional remainder'''
5 // 3


# Mutable and immutable objects
'''
Most objects in Python are mutable, such as lists, dicts, NumPy arrays, or most userdefined
types (classes).'''
a_list = ['foo',2,[4,5]]

a_list[2] = (3,4)

a_list

type(a_list[2])

'''
Others, like strings and tuples, are immutable:'''

'''
Python strings are immutable; you cannot modify a string without 
creating a new string:'''

a = 'this is a string'

b = a.replace('string', 'longer string')
b

'''
Strings are a sequence of characters and therefore can be treated like other 
sequences, such as lists and tuples:'''

s = 'python'
list(s)
tuple(s)
s[:3]


'''
The backslash character \ is an escape character, meaning that it is used to 
specify special characters like newline \n or unicode characters. 
To write a string literal with backslashes, you need to escape them:'''
s = '12\34'
print(s)

s1 = '12\\34'
print(s1)

'''
If you have a string with a lot of backslashes and no special characters, 
you might find this a bit annoying. 
Fortunately you can preface the leading quote of the string with r
which means that the characters should be interpreted as is:'''

s = r'this\has\no\special\characters'
print(s)


'''
Strings with a % followed by one or more format characters
is a target for inserting a value into that string.'''

template = '%.2f %s are worth $%d'
'''
In this string, %s means to format an argument as a string, %.2f a number with 2 decimal
places, and %d an integer. To substitute arguments for these format parameters, use the
binary operator % with a tuple of values:'''

template % (4.5560, 'Argentine Pesos', 1)



# Dates and times

from datetime import datetime, date, time

dt = datetime(2011,10,29,20,30,21)

dt.day
dt.minute

'''
Given a datetime instance, you can extract the equivalent date and time objects by
calling methods on the datetime of the same name:'''
dt.date()
dt.time()

'''
The strftime method formats a datetime as a string:'''
dt.strftime('%m/%d/%y %H:%M')

'''
Strings can be converted (parsed) into datetime objects using the strptime function:'''
datetime.strptime('20091031', '%Y%m%d')

'''
When aggregating of otherwise grouping time series data, it will occasionally be useful
to replace fields of a series of datetimes, for example replacing the minute and second
fields with zero, producing a new object:'''
dt.replace(minute=0,second=0)

'''
The difference of two datetime objects produces a datetime.timedelta type:'''
dt2 = datetime(2011,11,15,22,30)

delta = dt2 - dt
delta
type(delta)

'''
Adding a timedelta to a datetime produces a new shifted datetime:'''
dt

dt + delta == dt2



# for loops
'''
for loops are for iterating over a collection (like a list or tuple) or an 
iterater. The standard syntax for a for loop is:'''

for value in collection:
    # do something with value
    
'''
A for loop can be advanced to the next iteration, skipping the remainder of the block,
using the continue keyword. Consider this code which sums up integers in a list and
skips None values:'''
sequence = [1,2,None,4,None,5]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value

total

'''
A for loop can be exited altogether using the break keyword. This code sums elements
of the list until a 5 is reached:'''
sequence = [1,2,0,4,6,5,2,1]

total_until_5 = 0

for value in sequence:
    if value == 5:
        break
    total_until_5 += value

total_until_5

'''
As we will see in more detail, if the elements in the collection or iterator are sequences
(tuples or lists, say), they can be conveniently unpacked into variables in the for loop
statement:
for a, b, c in iterator:
    # do somthing
'''


# while
'''
A while loop specifies a condition and a block of code that is to be executed
until the condition evaluated to False or the loop is explicitly ended with 
break:'''

x = 256
total = 0
while x>0:
    if total > 500:
        break
    total += x
    x = x // 2

total
x

x = 256
total = 0
while total < 500:
    if x < 0 :
        break
    total += x
    x = x // 2



# Exception handling
def attempt_float(x):
    try:
        return float(x)
    except:
        return x
    
'''
The code in the except part of the block will only be executed if float(x) raises an
exception.'''

    

# range and xrange
list(range(0,20,2))

'''
a common use of range is for iterating through sequences by index:'''
seq = [1,2,3,4]
for i in range(len(seq)):
    val = seq[i]
    print(val)


# ternary expression
x = 5
'non-negative' if x>=0 else 'negative'



# Tuple
'''
A tuple is a one-dimensional, fixed-length, immutable sequence of Python objects.'''

tup = 4,5,6
tup

'''
Any sequence or iterator can be converted to a tuple by invoking tuple:'''
tuple([4,0,3])

tuple('stirng')


tup = tuple(['foo',[1,2],True])
tup[1].append(3)
tup

'''
If you try to assign to a tuple-like expression of variables, Python will attempt to unpack
the value on the right-hand side of the equals sign:'''

tup = (4,5,6)
a, b, c = tup
b

tup = 4,5,(6,7)
a,b,(c,d) = tup
c


a = (1,2,2,2,2,3,4,2)
a.count(2)



# List
'''
In contrast with tuples, lists are variable-length and their contents can be modified.'''


#Adding and removing elements

b_list = ['foo','bar','baz']
b_list.append('dwarf')

'''
Using insert you can insert an element at a specific location in the list:'''
b_list.insert(1,'red')
b_list

'''
The inverse operation to insert is pop, which removes and returns an element at a
particular index:'''
b_list.pop(2)
b_list

'''
Elements can be removed by value using remove, which locates the first such value and
removes it from the last:
'''
b_list.remove('foo')
b_list


'''
If you have a list already defined, you can append multiple elements to it using the
extend method:'''
x = [4,None,'foo']
x.extend([7,8,(2,3)])
x
x.enxtend([5])


# Sorting
a = [7,2,5,1,3]

a.sort()
a

b = ['saw', 'small', 'He', 'foxes', 'six']

b.sort(key=len)
b


# slicing

seq = [7, 2, 3, 6, 3, 5, 6, 0, 1]

'''
A step can also be used after a second colon to, say, take every other element:'''
seq[::2]

'''
A clever use of this is to pass -1 which has the useful effect of reversing a list or tuple:'''
seq[::-1]



# Built-in Sequence Function

# enumerate
'''
It’s common when iterating over a sequence to want to keep track of the index of the
current item. A do-it-yourself approach would look like:
i = 0
for value in collection:
    # do something with value
    i += 1

Since this is so common, Python has a built-in function enumerate which returns a
sequence of (i, value) tuples:
for i, value in enumerate(collection):
    # do something with value

'''

'''
When indexing data, a useful pattern that uses enumerate is computing a dict mapping
the values of a sequence (which are assumed to be unique) to their locations in the
sequence:'''
some_list = ['foo','bar','baz']

mapping = dict((v,i) for i,v in enumerate(some_list))
mapping


# sorted
'''
A common pattern for getting a sorted list of the unique elements in a sequence is to
combine sorted with set:'''
sorted(set('this is just some string'))


# zip
'''
zip “pairs” up the elements of a number of lists, tuples, or other sequences, to create
a list of tuples:'''
seq1 = ['foo', 'bar', 'baz','c']
seq2 = ['one', 'two', 'three']

list(zip(seq1,seq2))

'''
zip can take an arbitrary number of sequences, and the number of elements it produces
is determined by the shortest sequence:'''
seq3 = [False, True]

list(zip(seq1,seq2,seq3))


''' zip's common use: simultaneously iterating over multiple sequences
'''
for i, (a,b) in enumerate(zip(seq1,seq2)):
    print('%d: %s, %s' % (i,a,b))


    
'''
Given a “zipped” sequence, zip can be applied in a clever way to “unzip” the sequence.
Another way to think about this is converting a list of rows into a list of columns.
'''
pitchers = [('Nolan','Ryan'), ('Roger','Clemens'), ('Schilling','Curt')]

first_names, last_names = zip(*pitchers)
print(first_names)
print(last_names)

list(zip(*pitchers))

list(zip(pitchers[0],pitchers[1],pitchers[2]))


# reversed
list(reversed(range(10)))



# Dict
'''
dict is likely the most important built-in Python data structure. A more common name
for it is hash map or associative array. It is a flexibly-sized collection of key-value pairs,
where key and value are Python objects.'''

d1 = {'a':'some value', 'b':[1,2,3,4]}

d1[7] = 'an interger'
d1

print(d1['b'])

'''
You can check if a dict contains a key using the same syntax as with checking whether
a list or tuple contains a value:'''

'b' in d1


'''
Values can be deleted either using the del keyword or the pop method (which simultaneously
returns the value and deletes the key):'''

d1[5] = 'some value'
d1['dummy'] = 'another value'
d1

del d1[5]

d1.pop('dummy')
d1

d1.keys()
d1.values()

'''
one dict can be merged into another using the update method.
'''

d1.update({'b':'foo','c':12})
d1


# creating dicts from sequences
'''
It’s common to occasionally end up with two sequences that you want to pair up element-
wise in a dict.

As a first cut, you might write code like this:

mapping = {}
for k, value in zip(key_list, value_list):
    mapping(k) = value

Since a dict is essentially a collection of 2-tuples, it should be no shock that the dict
type function accepts a list of 2-tuples:
'''

mapping = dict(zip(range(5), reversed(range(5)))


# default values
'''
value = some_dict.get(key, default_value)

For example, you could imagine categorizing a list of words by their first letters
as a dict of lists:
'''
words = ['apple', 'bat', 'bar', 'atom', 'book']

by_letter = {}

for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)

by_letter


# Valid dict key types
'''
While the values of a dict can be any Python object, the keys have to be immutable
objects like scalar types (int, float, string) or tuples (all the objects in the tuple need to
be immutable, too). The technical term here is hashability. You can check whether an
object is hashable (can be used as a key in a dict) with the hash function:
'''
hash('string')

hash((1,2,(2,3)))

hash((1,2,[2,3])) # fails because lists are mutable

'''    
To use a list as a key, an easy fix is to convert it to a tuple:'''

d = {}
d[tuple([1,2,3])] = 5
d

{'a':5,'a':6,7:'y'}
# keys in dicts must be unique

        

# Set
'''
A set is an unordered collection of unique elements.
You can think of them like dicts, but keys only, no values.
'''
set([2,2,2,1,3,3])    #{1,2,3}

{2,2,2,1,3,3}          #{1,2,3}


'''
Sets support mathematical set operations like union, intersection, difference, and symmetric
difference.
'''
a = {1,2,3,4,5}
b = {3,4,5,6,7,8}

a | b  #union(or)

a & b  #intersection(and)

a - b  #difference

a ^ b  #symmetric difference: all of the elements in a or b but not both

'''
check if a set is a subset or a superset of another set
'''
a_set = {1,2,3,4,5}

{1,2,3}.issubset(a_set)

a_set.issuperset({1,2,3})


'''
As you might guess, sets are equal if their contents are equal:'''

{1,2,3} == {3,2,1}



# List, Set, and Dict Comprehensions
'''
List comprehensions allow you to concisely form a new list by filtering the 
elements of a collection and transforming the elements passing the filter in 
one conscise expression. They take the basic form:

   [expr for val in collection if condition]

For example, given a list of strings, we could filter out strings with length 2 
or more and also convert them to uppercase like this:'''

strings = ['a', 'as', 'bat', 'car', 'dove', 'python']

[x.upper() for x in strings if len(x)>=2]

'''
A dict comprehension and a set comprehension look like:
    
    dict_comp = {key-expr: value-expr for key,value in collection if condition}
    
    set_comp = {expr for value in collection if condition}'''

'''
Suppose we wanted a set containing just the lengths of the strings contained in the
collection; this could be easily computed using a set comprehension:'''
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']

{len(x) for  x in strings}

'''
As a simple dict comprehension example, we could create a lookup map of these strings
to their locations in the list:'''
{index: val for index,val in enumerate(strings)}


# Nested list comprehension

all_data=[['Tom','Billy','Jefferson','Andrew','Wesley','Steven','Joe'],
          ['Susie','Casey','Jill','Ana','Eva','Jennifer','Stephanie']] 

'''
Supppose we want to get a single list containing all names with two or more 
e's in them.
'''

# solution 1
names_of_interest = []

for names in all_data:
    for name in names:
        if name.count('e') >= 2:
            names_of_interest.append(name)

# solution 2
names_of_interest = []

for names in all_data:
    enough_es = [name for name in names if name.count('e')>=2]
    names_of_interest.extend(enough_es)

# solution 3 （nested list comprehension
result = [name for names in all_data  for name in names
          if name.count('e')>=2]
result


'''
Here is another example where we 'flatten' a list of tuples of integers 
into a simle list of integers:'''
some_tuples = [(1,2,3),(4,5,6),(7,8,9)]

flattened = [x for tup in some_tuples  for x in tup]
flattened










        
# Functions

# Functions are objects
states = [' Alabama','Georgia!','Georgia','georgia','FlOrIda',
          'south  carolina##','West virginia?']
'''
data cleaning:
whitespace stripping, removing punctuation symbols, 
and proper capitalization
'''

import re

def clean(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]','',value)
        value = value.title()
        result.append(value)
    return result

clean(states)


'''
An alternative approach is to make a list of the operations you want to apply to 
a particular set of strings.
'''
import re

def remove_punctuation(value):
    return re.sub('[!#?]','',value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

clean_strings(states, clean_ops)


# map function, which applies a function to a collection of some kind
list(map(remove_punctuation, states))
'''
[' Alabama',
 'Georgia',
 'Georgia',
 'georgia',
 'FlOrIda',
 'south  carolina',
 'West virginia']
'''



# Anonymous (lambda) functions
'''
Anonymous or lambda functions are really just simple functions 
consisting of a single statements, the result of which is the return value.
'''
equiv_anon = lambda x: x * 2
equiv_anon(5)   #10

'''
It's often less typing (and clearer) to pass a lambda function 
to a local variable
'''
def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

ints = [4,0,1,5,6]
apply_to_list(ints, lambda x: x*2)
#[8, 0, 2, 10, 12]

strings = ['fool','card','bar','aaaa','abab']
'''
sort a collection of strings by the number of distinct 
letters in each string
'''
strings.sort()
strings
#['fool','card','bar','aaaa','abab']

strings.sort(key= lambda x: len(x))
#['bar', 'fool', 'card', 'aaaa', 'abab']

strings.sort(key= lambda x: len(set(list(x))))
#['aaaa', 'abab', 'fool', 'bar', 'card']


# Closures: functions that return functions
'''
In a nutshell, a closure is any dynamically-generated function 
returned by another function.
The returned function has access to the variable in the local 
namespace where it was created.
'''
def make_closure(a):
    def closure():
        print('I know the secret: %d' % a)
    return closure

closure = make_closure(5)
list(closure)

# Example2
def make_watcher():
    have_seen = {}
    
    def has_been_seen(x):
        if x in have_seen:
            return True
        else:
            have_seen[x] = True
            return False
    
    return has_been_seen

watcher = make_watcher()

vals = [5,6,1,5,1,6,3,5]

[watcher(x) for x in vals]
#[False, False, False, True, True, True, False, True]

'''
While you can mutate any internal state objects (like adding key-value 
pairsto a dict), you cannot bind variables in the enclosing dunction scope.
One way to work around this is to modify a dict or list ranther than 
binding variables:
'''
def make_counter():
    count = [0]
    def counter():
        # invrement and return the current count
        count[0] += 1
        return count[0]
    return counter

counter = make_counter


# Example3
'''
Creating a string formatting function
'''
def format_and_pad(template, space):
    def formatter(x):
        return (template % x).rjust(space)
    
    return formatter

fmt = format_and_pad('%.4f',15)
fmt(1.756)
# '         1.7560'


# Extended Call Syntax with *args, **kwargs
def say_hello_then_call_f(f, *args, **kwargs):
    print('args is', args)
    print('kwargs is', kwargs)
    print("Hello! Now I'm going to call %s" % f)
    return f(*args, **kwargs)

def g(x, y, z=1):
    return (x+y)/z

say_hello_then_call_f(g, 1, 2 ,z=5)
#args is (1, 2)
#kwargs is {'z': 5}


# Currying: Partial Argument Application
'''
Currying means deriving new functions from existing ones by partical
argument application.
'''
def add_numbers(x,y):
    return x+y

add_five = lambda y: add_numbers(5, y)

add_five(4)     #9

'''
The built-in functools module cansimplify this process 
using the partial function:
'''
from functools import partial
add_five = partial(add_numbers, 5)

add_five(5)   #10

'''
When discussing pandas and time series data, we'll use this technique to 
create specialized functions for transforming data series
'''
# compute 60-day moving average of time series x
#ma60 = lambda x: pandas.rolling_mean(x, 60)
# Take the 60-day moving average of all time series in data
#data.apply(ma60)


# Generators
''' 
the iterator protocol:
    a generic way to make objects iterable.
'''
some_dict = {'a':1, 'b':2, 'c':3}

for key in some_dict:
    print(key)
#a b c

dict_iterator = iter(some_dict)
dict_iterator
#<dict_keyiterator at 0x2cec2c8dd68>

'''
Any iterator is any object that will yield objects to the 
Python interpreter when used in a context like a for loop.
Most methods expecting a list or list-like object will also
accept any iterable object. This includes built-in methods such
as min, max, and sum, and type constructors like list and tuple:
'''
list(dict_iterator)
#  ['a', 'b', 'c']

'''
A generator is a simple way to construct a new iterable object.
Whereas normal functions execute and return a single value, generators
return a sequence of values lazily, pausing after each one until 
the next one is required. 
To create a generator, use the yield keyword instead of return in a function.
'''
def squares(n = 10):
    print('Generating squares from 1 to %d' % (n**2))
    for i in range(1, n+1):
        yield i**2

gen = squares()

gen
#<generator object squares at 0x000002CEC2BCDBA0>
'''
It's not until you request elements from the generator that 
it begins executing its code.
'''
for x in gen:
    print(x)

# Generating squares from 1 to 100
#1 4 9 16 25 36 49 64 81 100

# Comparing to 
def squares(n = 10):
    print('Generating squares from 1 to %d' % (n**2))
    for i in range(1, n+1):
        return i**2

gen = squares()

gen    # 1


# generator expressions 
'''
A simple way to make a generator is by using a generator expression.
To create one, enclose what would otherwise be a list comprehension
with parenthesis instead of brakckets.
'''
gen = (x ** 2 for x in range(100))

gen
#<generator object <genexpr> at 0x000002CEC2CA7DB0>

'''
This is completely equivalent to the following more verbose generator.
'''
def _make_gen():
    for x in range(100):
        yield x ** 2
gen = _make_gen()

'''
Generator expressions can be used inside any Python function that will
accept a generator.
'''
sum(x**2 for x in range(100))   #328350

dict((i, i**2) for i in range(5))
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# itertools module
'''
The standard library itertools module has a collection of generators
for many common data algorithms.
For example, groupby take any sequence and a function; this groups 
consecutive elements in the sequence by return value of the function.
'''
import itertools
first_letter = lambda x : x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']

for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names))  # names is a generator
# A ['Alan', 'Adam']
# W ['Wes', 'Will']
# A ['Albert']
# S ['Steven']

        