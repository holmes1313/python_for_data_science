
import numpy as np

# 4.1 The NumPy ndarray: a multidimensional arrary object
'''
An ndarray is a generic multidimensional container for same type data, that is,
all of the elements must be the same type.'''


## 4.1.1 Creating ndarrays
'''
The array function accepts any sequence-like object (including other arrays).
'''
data1 = [6,7.5,8,0,1]
arr1 = np.array(data1)
arr1
arr1.shape
arr1.dtype

'''
Nested sequences, like a list of equal-length lists, will be converted into
a multidimensional array'''
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
arr2
arr2.shape
arr2.dtype
arr2.ndim   # 2

'''
data.shape: a tuple indicating the size of each dimension
data.dtype: an object describing the data type of the array
'''
arr2.shape   # (2, 4)
arr1.dtype   # dtype('float64')
arr2.dtype   # dtype('int32')

'''
In addition to np.array, there are a number of other functions for creating 
new arrays.
np.zeros() and np.ones() create arrays of 0's or 1's, respectively, with
a given length or shape.
np.empty() creates any array without initializing its values to any particular
value.
To create a higher dimensional array with theses methods, pass a tuple for 
the shape:'''
np.zeros(10)

np.ones((3,6))

np.empty((3,2,1))

'''
np.arange is an array-valued version of the built-in Python range function:'''
np.arange(10).reshape(2,5)


## 4.1.2 Data type for ndarrays
'''
The data type or dtype is a special object containing the information the 
ndarray needs to interpret a chunk of memory as a particular type of data:'''
np.array([1,2,3],dtype='float')
np.array([1.1,2.2,3.3], dtype='int')
np.array([2,3,4],dtype='object')

'''
You can explicitly convert or cast aan array from one dtype to another using 
ndarray's astype method:'''
arr = np.array([1,2,3,4,5])
arr.dtype

arr.astype('float')
arr.astype('str')


'''
In this example, integers were cast to floating point. If I cast some floating
point numbers to be of integer dtype, the decimal part will be truncated.'''
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr.astype(np.int32)
'''
Calling astype always creates a new array (a copy of the data), even if 
the new dtype is the same as the old dtype.'''

'''
Should you have an array of strings representing numbers, you can use astype
to convert them to numeric form:'''
numeric_strings = np.array([1.25, -9.6, 42], dtype=np.string_)
numeric_strings
numeric_strings.astype(np.float64)


'''
You can also use another array's dtype attribute:'''
int_array = np.arange(10)
calibers = np.array([.22, .270, .357], dtype=np.float64)

int_array.astype(calibers.dtype)


## 4.1.3 Operations between arrays and scalars
'''
Arrays enable you to express batch operations on data without writting any
for loops. This is usually called vectorization. 
Any arithmetic operations between equal-size arrays applies 
the operaton elementwise:'''
arr = np.array([[1,2,3],[4,5,6]])
arr
arr * arr
arr - arr

'''
Arithmetic operations with scalars are propagating the value to each element:'''
2 * arr
1 / arr

'''
Operations between differently sized arrays is called broadcasting and will be 
discussed in more detail in Chapter 12.'''


## 4.1.4 Basic Indexing and slicing
'''
There are many ways you may want to select a subset of your data or individual 
elements.'''
arr = np.arange(10)
arr

arr[5]

arr[5:8]

arr[5:8] = 12
arr

'''
If you assign a scalar value to a slice, as in arr[5:8]=12, the value is propagated
(or broadcasted henceforth) to the entire selection.

An important first discussion from lists is that array slices are views on 
the original array, so any modifications to the view will be reflected in 
the source array.'''
arr_slice = arr[5:8]
arr_slice
arr_slice[1] = 12345
arr

arr_slice
arr_slice[:] = 64
arr

'''
If you want a copy of a sice of an ndarray instead of a view, you will need 
to explicitly copy the array: arr[5:8].copy().'''
arr[5:8].copy()

'''
With higher dimensional arrays, you have many more options.'''
arr2d = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
arr2d[2]

'''
Thus, individual elements can be accessed recursively. But that is a bit too 
much work, so you can pass a comma-sparated list of indices to select individual
elements.'''
arr2d[0][2]
arr2d[0,2]


'''
In multidimensional arrays, lie 2*2*3 array arr3d: i'''
arr3d = np.array([[[1,2,3],
                   [4,5,6]],
                  
                  [[7,8,9],
                   [10,11,12]]])
arr3d
arr3d[0]   # 2*3

'''
Both scalar values and arrays can be assigned to arr3d[0]:'''
old_values = arr3d[0].copy()
old_values

arr3d[0] = 42
arr3d

arr3d[0] = old_values
arr3d


# Indexing with slices
'''
Higher dimensional objects give you more options as you can slice one or more
axes and also mix integers.'''
arr2d = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
arr2d[:2]
arr2d[:2,1:]

'''
By mixing integer indexes and slices, you get lower dimensional slices:'''
arr2d[1,:2]
arr2d[2,:1]

'''
Note that a colon by itself means to take the entire axis, so you can slice only
higher dimensional axes by doing:'''
arr2d[:,:1]
arr2d[:,0]

'''
Of course, assigning to a slice expression assigns to the whole selection:'''
arr2d[:2, 1:]
arr2d[:2, 1:] = 0
arr2d

## 4.1.5 Boolean indexing
'''
We have some data in an array and an array oof names with duplicates.'''
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])

'''
the randn function in numpy.random can generate some random normally 
distributed data.'''
data = np.random.randn(7,4)
data

'''
Suppose each name corresponds to a row in the data array and we wanted to select
all the rows with corresponding name 'Bob'. 

Like arithmatic operations, comparisons (such as ==) with arrays are also 
vectorized.'''
names == 'Bob'

'''
This boolean array can be passed when indexing the array:'''
data[names == 'Bob']   # Only pick line0 and line3

'''
The boolean array must be of the same length as the axis it's indexing.
You can even mix and match boolean arrays with slices or integers (or sequences
of integers, more on this laters):'''
data[names == 'Bob', 2:]
data[names=='Bob',3]
data[names=='Bob',3:]

'''
To select everything but 'Bob', you can either use != or negatethe condition using
-'''
names != 'Bob'
data[-(names == 'Bob')]


'''
Selecting two of the three names to combine multiple boolean considitions, 
use boolean arithmatic operators like &(and) and |(or):'''
mask = (names == 'Bob') | (names == 'Will')
mask
data[mask]
'''
Selecting data from an array by boolean indexing always creates a copy of the data,
even if the returned array is unchanged.'''

'''
Setting values with boolean arrays works in a common-sense way.
To set all of the negative values in data to 0'''
data
data < 0
data[data<0] = 0
data


## 4.1.6 Fancy indexing
'''
Fancy indexing is a term adopted by Numpy to describe indexing using integer 
arrays.'''
arr = np.empty((8,4))
arr

for i in range(8):
    arr[i] = i
arr

'''
To select out a subset of the rows in a particular order, just pass
a list or ndarray of integers specifying the desired order:'''
arr[[4,3,0,6]]
arr[[-3, -5, -7]]

'''
Passing multiple index arrays selects a 1D array of elements 
corresponding to each tuple of indices:'''
arr = np.arange(32).reshape(8,4)
arr
arr[[1,5,7,2],[0,3,1,2]]
arr[1,0]

'''
array([ 4, 23, 29, 10])
the elements (1,0), (5,3), (7,1), (2,2) were selected.'''

'''
The behavior of fancy indexing is the rectangular region formed by
selecting a subset of the matrix's rows and columns.'''
arr
arr[[1,5,7,2]]
arr[[1,5,7,2]][:, [0,3,1,2]]
   
'''
Another way is to use the np.ix_function, which converts two 1D integer
arrays to an indexer that selects the square region:'''
arr
arr[np.ix_([1,5,7,2],[0,3,1,2])]

'''
Keep in mind that fancy indexing, unlike slicing, always copies the data into 
a new array.'''    


## 4.1.7 Tranposing arrays and swapping axes
'''
Tranposing is a special form of reshaping which similarly returns a view on
the underlying data without copying anything.
Arrays have the transpose method and also the special T attribute:'''
arr = np.arange(15).reshape((3,5))
arr    
    
arr.T
arr

'''
When doing matrix computations, you will do this oftern, like for example
computing the inner matrix product XT*X using np.dot:'''
arr.dot(arr.T)
np.dot(arr.T, arr)
np.dot(arr, arr.T)

'''
Simple tranposing with .T is just a special of swapping axes, ndarray has the
method swapaxes which takes a pair of axis numbers:'''
arr
arr.swapaxes(1,0)
arr.T



# 4.2 Universal functions: fast element-wise array functions
'''
A universal function, or ufunc, is a function that performs elementwise
operations on data in ndarrays.
You can think of them as fast vectorized wrappers for simple functions that take
one or more scalar values and produces one or more scalar results.'''
arr = np.arange(10)  
arr

np.sqrt(arr)
arr
    
np.exp(arr)    

'''
np.sqrt and np.exp are referred to as unary ufuncs.
Others, such as add or maximum, takes 2 arrays (thus, binary ufuncs) and
return a single array as the result:'''
x = np.random.randn(8)
y = np.random.randn(8)
x
y

np.maximum(x,y)    # element-wise maximum
np.max(x)
    
'''
While not common, a ufunc can return multiple arrays.
modf is one example, a vectorized version of the built-in Python divmod: 
it returns the fractional and integral parts of a floating point array:'''
arr = np.random.randn(7) * 5
arr
np.modf(arr)[1]
   
# Unary ufuncs 
a = np.random.randn(4)
a
np.abs(a)

np.sqrt(np.square(a))
np.square(a)

np.exp(a)
np.log(np.exp(a))

np.sign(a)

np.ceil(a)
np.floor(a)
np.rint(a)
np.modf(a)

np.isnan(a)

    
# 4.3 Data processing using arrays
'''
evaluate the function sqrt(x^2 + y^2) across a regular grid of values.
The np.meshgrid function takes two 1D arrays and produces two 2D matrics 
corresponding to all pairs of (x,y) in the two arrays:'''
points = np.arange(-5,5,0.1)  # 100 equally spaced points
points

xs, ys = np.meshgrid(points, points)
xs
ys
    
import matplotlib.pyplot as plt
    
z = np.sqrt(xs**2 + ys**2)    
z

plt.imshow(z,cmap=plt.cm.gray); plt.colorbar()

plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
    

## 4.3.1 Expressing conditional logic as array operations
'''
The np.where function is a vectorized version of the ternary expression
x if condition else y.'''

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

'''
Suppose we wanted to take a value from xarr whenever the corresponding value
in cond is True otherwise take the value from yarr.'''

''' 
A list comprehension doint this might look like:'''
list(zip(xarr, yarr, cond))

result = [(x if c else y)
            for x, y, c in zip(xarr, yarr, cond)]
result

'''
With np.where you can write this very concisely'''
result = np.where(cond, xarr, yarr)
result     # array([ 1.1,  2.2,  1.3,  1.4,  2.5])

'''
The second and third arguments to np.where don't need to be arrays;
one or both of them can be scalars.
A typical use of where in data analysis is to product a new array of values
based on another array.'''

'''
Suppose you had a matrix of randomly generated data and you wanted to replace
all positive values with 2 and all negative values with -2.'''
arr = np.random.randn(4, 4)
arr

np.where(arr>0, 2, -2)

np.where(arr>0, 2, arr) # set only positive values to 2
# or
arr[arr>0]  = 2
arr

'''
with some cleverness you can use where to express more complicated logic:
    
result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)

        
This for loop can be converted into a nested where expression:
np.where(cond1 & cond2, 0,
         np.where(cond1, 1,
                  np.where(cond2, 2, 3)))

We can also take advantage of the fact that boolean values are treated as 0 or
1 in calculations.

result = 1*(cond1 & -cond2) + 2*(-cond1 & cond2) + 3*-(cond1 | cond2)
'''

## 4.3.2 Mathematical and statistical methods
arr = np.random.randn(5,4)  # normally-distributed data
arr

arr.mean()
# or
np.mean(arr)

arr.sum()
# or 
np.sum(arr)

arr.std()
# or 
np.std(arr)

'''
Functions like mean and sum take an optional axis argument which computes
the statistic over the given axis'''
arr
arr.mean(axis=1)   # rows

arr.sum(axis=0)   # columns

'''
Other methods like cumsum and cumprod do not aggregate, instead producing
an array of the intermediate resultes:'''
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr

arr.cumsum()
arr.cumsum(axis=0)  # columns
arr.cumsum(axis=1)

arr.cumprod()
arr.cumprod(axis=1)
arr.cumprod(axis=0)


## 4.3.3 methods for Boolean arrays
'''
Boolean values are coerced to 1 (True) and 0 (False). Thus, sum is often 
used as a means of counting True values in a boolean array:'''
arr = np.random.randn(100)
(arr>0).sum()      # Numbers of positive values

'''
any tests whether one or more values in an array is True, while all checks
if every value is True:'''
bools = np.array([False, False, True, False])
bools.any()   #True
bools.all()   #False

'''
These methods also work with non-boolean arrays, where non-zero elements
evaluate to True.'''
tt = np.array([0,-1])
tt.any()
tt.all()


## 4.3.4 Sorting
'''
Numpy arrays can be sorted in-place using the sort method:'''
arr = np.random.randn(8)
arr

np.sort(arr)
sorted(arr)
arr.sort()
arr


'''
Multidimensional arrays can have each 1D section of values sorted in-place
along an axis by passing the axis number to sort:'''
arr = np.random.randn(5,3)
arr

arr.sort(1)  # sort each row
arr

arr.sort(0)   # sort each column
arr


'''
The function np.sort returns a sorted copy of an array instead of 
modifying the array in place.'''
arr = np.random.randn(5,3)
arr

np.sort(arr)
np.sort(arr, axis=1)
arr

'''
A quick-and-dirty way to compute the quantitles of an array is to
sort it and select the value at a particular rank:'''
large_arr = np.random.randn(1000)
large_arr

large_arr.sort()
large_arr
int(0.05 * len(large_arr))
large_arr[int(0.05 * len(large_arr))]  # 5% quantile

np.sort(large_arr)[int(0.05 * len(large_arr))]


## 4.3.5 Unique and other set logic    
'''
np.unique returns the sorted unique values in an array:'''
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])    
np.unique(names)    

ints = np.array([3,3,3,2,2,1,1,4,4])
np.unique(ints)

'''
Contrast np.unique with the pure Python alternative:'''
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])    
set(names) 

sorted(set(names))
np.sort(np.unique(names))
names


'''
another function, np.in1d,  tests membership of the values in one array in another, 
returning a boolean array:'''
values = np.array([6,0,0,3,2,5,6])

np.in1d(values, [2,3,6])    

    
# 4.4 File input and output with arrays

## 4.4.1 Sorting arrays on disk in binary format
'''
np.save and np.load are the two workhorse functions for efficiently saving and
loading array data on disk. 
Arrays are saved by default in an uncompressed raw binary formate with file 
extension .npy.'''

arr = np.arange(10)
np.save('some_array',arr)


'''
If the file path does not already end in .npy, the extension will be appended.
The array on disk can then be loaded using np.load:'''

np.load('some_array.npy')


'''
You save multiple arrays in a zip archive using np.savez and passing the arrays
as keyword arguments:'''
np.savez('array_archive.npz',a=arr, b=arr)

'''
When loading an .npz file, you get back a dict-like object which loads the 
individual arrays lazily:'''
arch = np.load('array_archive.npz')
arch['b']

## 4.4.2 Saving and loading text file
# arr = np.loadtxt('array_ex.txt', delimiter=',')



# 4.5 Linear algebra
'''
Unlike some other languages, multiplying two two-dimensinal arrays with * is an
element-wise product instead of a matrix dot product.
There is a function dot, both an array method, and a function in the numpy 
namespace, for matrix multiplication:'''

x = np.array([[1,2,3], 
              [4,5,6]])
y = np.array([[6,23],
              [-1,7],
              [8, 9]])
y    
y.T

x * y.T 

x.dot(y)    # equivalently np.dot(x,y)

'''
A matrix product between a 2D array and a suitable sized 1D arra results in
a 1D array:'''
np.ones(3)    #array([ 1.,  1.,  1.])
np.dot(x,np.ones(3))

np.dot(np.ones(3),x)
# ValueError: shapes (3,) and (2,3) not aligned: 3 (dim 0) != 2 (dim 0)

'''
numpy.linalg has a standard set of matrix decompositions and things like inverse
and determinant.'''

# inverseand determinant
from numpy.linalg import inv, qr

x = np.random.randn(5,5)

mat = x.T.dot(x)

inv(mat)

mat.dot(inv(mat))

q, r = qr(mat)

r


# 4.6 Random Number generation
'''
The numpy.random module supplements the built-in python random with functions
for efficiently generating whole arrays of sample values from many kinds of probability
distributions.'''

'''
To get a 4 by 4 array of samples from the standard normal distribution using 
normal:'''
samples = np.random.normal(size=(4,4))
samples

'''
Table 4-8 for a partial list of functions available in numpy.random
rand: a uniform dist
randint: random integers from a given low-to-high range
randn: standard normal dist 
'''


# 4.7 Example: random walks
'''
An illustrative application of utilizing array operations is in the simulation
of random walks.
a simple random walk starting at 0 with steps of 1 and -1 occuring with equal
probability.'''
import numpy as np
position = 0
walk = [position]
steps = 100
for i in range(steps):
    step = 1 if np.random.randint(0,2) else -1
    position += step
    walk.append(position)
walk

'''
use the np.random module to draw 1,000 conin flips at once, set these to 1 and
-1, and compute the cumulative sum:'''
nsteps = 100

draws = np.random.randint(0,2,size=nsteps)
draws

steps = np.where(draws>0, 1, -1)
steps

walk = steps.cumsum()
walk

walk.min()
walk.max()

# first crossing time
(np.abs(walk) >= 10).argmax()
'''
The first crossing time: the step at which the random walk reaches a 
particular value.
argmax returns the first index of the maximum value in the boolean array (True
is the maximum value):'''
np.abs(walk)>= 10

(np.abs(walk)>= 10).argmax()  

(walk < -10).argmax()
  
  
# Similation Many Random Walks at Once
nwalks = 500

steps = 100

draws = np.random.randint(0,2, size=(nwalks, steps))    
draws.shape    

steps = np.where(draws>0, 1, -1)    
steps    
'''
draws
draws[draws==0] = -1
draws'''

walks = np.cumsum(steps, axis=1)    
walks 

walks.max(axis=1)    
walks.min(axis=1)    

hits10 = (np.abs(walks) >= 10).any(axis=1) 

hits10.sum()    

crossing_times = (abs(walks[hits10])>=10).argmax(axis=1)    
crossing_times.mean()    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
