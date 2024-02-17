# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:30:59 2023

@author: santo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:26:55 2023

@author: yuvra_d0x8avj
"""

'''
Numpy: it is the open source library ,
        it is used for specific computing application
        It is also stand for numerical python
        consisting of multidimensional array
        Objects and collection of routines for processsing the


'''
# one dimensional array using numpy

import numpy as np
arr=np.array([10,20,30])
print(arr)
print(type(arr))

# multi dimensional array using numpy

 # 2D  array
import numpy as np
arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
    # accessing 
for i in arr:
    for j in i:
        print(j,end=' ')

 # 3D array
import numpy as np
arr=np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]])

for i in arr:
    for j in i:
        for k in j:
            print(k,end=' ')
        print()
   

 # ndim : we can provide dimension to array

import numpy as np

arr=np.array([1,2,3,4,5],ndmin=3)
arr


arr=np.array([1,2,3,4,5],ndmin=2)
arr

 # change the datatype

import numpy as np
arr=np.array([1,2,3,4,5],dtype=complex) # this will add imaginary part to code
arr

    # find dimension of array
import numpy as np
arr=np.array([[1,2,3,4],[5,6,7,8]])
print(arr.ndim)

    
    # find itemsize
    
import numpy as np
arr=np.array([[1,2,3,4],[5,6,7,8]])
print(arr.itemsize)

    # Data type 

import numpy as np
arr=np.array([11,12,13,14,15])
print(arr.dtype)

    # Get shape and size of array

import numpy as np

arr=np.array([[21,22,23,24,25],[26,27,28,29,30]])
print(arr.shape) # this give (row,columns)
print(arr.size) # gives total like row*column
print(arr[1,1])
print(arr[0,1])
print(arr[1,-1])

    # arrange(start,end,step)
    
import numpy as np
arr=np.arange(0,20,3) # create sequence of integer
print(arr)

arr1=np.arange(11) # create single element 
print(arr1)

print(arr1[1]) # access from left to right
print(arr1[-2]) # access from right to left

# access array element using slicing 
arr=np.array([1,2,3,4,5,6,7,8,9])
x=arr[1:8:2] # start:end:step of 2
x
'''
Output:
    
        array([2, 4, 6, 8])
'''

x=arr[-1:3:-1] #start from -1 goes upto 3 with step -1 but not include 3
x
'''
Output:
    
        array([9, 8, 7, 6, 5])

'''

# Slicing array:
    
    #for multi array
arr=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])

print(arr[1,2])
#8
print(arr[1,:])
#[ 6  7  8  9 10]
print(arr[:,1])
#[ 2  7 12]
x=arr[:3,:2]
print(x) 
'''
[[ 1  2]
 [ 6  7]
 [11 12]]
'''

y=arr[:3,::2] # rows from 0 to 3 and all alternate column
print(y)
'''
[[ 1  3  5]
 [ 6  8 10]
 [11 13 15]]
'''

# arrange and reshape
arr=np.arange(35).reshape(5,7) # it create array of 5 by 7 with 0 to 34 no
arr
'''
array([[ 0,  1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12, 13],
       [14, 15, 16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25, 26, 27],
       [28, 29, 30, 31, 32, 33, 34]])
'''

# boolean array indexing

    
arr=np.arange(12).reshape(3,4)
print(arr)
row=np.array([False,True,True]) # this means we dont  want 0th row and want only 1st and 2nd row
wanted_rows=arr[row,:]
print(wanted_rows)

# tolist : it will convert array to list

array=np.array([10,20,30,40])
print(type(array))

lst=array.tolist()
print(lst)
print(type(lst))

# Convert multidimensional array to list

marray=np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
y=marray.tolist()
print(y)
print(type(y))

# convert pyhton list to numpy array

    #numpy.array()
    #numpy.asarray()
    
lst1=[10,20,30,40]

x=np.array(lst1)

print('by array:',x)
print('Type is:',type(x))

y=np.asarray(lst1)

print('Type is:',type(y))
print('by asrray:',y)
                 

# Properties

arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])

    #shape
print(arr.shape)

    #size
print(arr.size)

    #reshape    
arr=np.array([[1,2,3],[6,7,8]])
print(arr.reshape(3,2))
 

###############################################################################
# Arithmatic Operation:
    
    #Apply arithmatic operations on numpy arrays
    
arr1=np.arange(16).reshape(4,4)
arr1
arr2=np.array([1,2,3,4])
arr2
arr=arr1+arr2
print(arr)
'''
[[ 1  3  5  7]
 [ 5  7  9 11]
 [ 9 11 13 15]
 [13 15 17 19]]
       
'''

arr=arr1-arr2
print(arr)
'''
[[-1 -1 -1 -1]
 [ 3  3  3  3]
 [ 7  7  7  7]
 [11 11 11 11]]
'''

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:05:06 2023

@author: yuvra_d0x8avj
"""
###############################################################################

# Write Numpy Version
import numpy as np
print(np.__version__)

###############################################################################

# write numpy program to check whether none of the element of array is zero

                              # all

import numpy as np

arr=np.array([1,2,3,4])
print('Original Array:',arr)
print('Is array is not containing zero?',np.all(arr)) 

arr2=np.array([1,2,3,4,0])
print('Original Array:',arr2)
print('Is array is not containing zeros?',np.all(arr2)) 


###############################################################################

# write numpy program to test if any of the element of array is non zero

                                # any

import numpy as np
arr3=np.array([1,2,3,0])
print(np.any(arr3))

import numpy as np
arr4=np.array([0,0,0,0,0])
print(np.any(arr4))

###############################################################################

# write numpy program to test given array element-wise for finiteness
# not infinity or not a number

                            # isfinite()

import numpy as np
arr1=np.array([1,0,np.nan,np.inf])
print('Test a given array element-wise for finiteness')
print(np.isfinite(arr1))

###############################################################################

# Write a numpy program to test element-wise NaN for a givena rray

                            # .isnan()
                            
import numpy as np

a=np.array([1,0,np.nan,np.inf])
print(np.isnan(a))


###############################################################################

# Write Numpy program to create an element-wise comparision
# (greater, greater_equal,less and less_equal)
 
                            #greater()
                            #greater_equal()
                            #less()
                            #less_equal()

x=np.array([3,5])
y=np.array([2,5])
print('Original numbers: ')
print(x)
print(y)
print('For Greater:')
print(np.greater(x,y))
print('For Greater_equal:')
print(np.greater_equal(x,y))
print('Less:')
print(np.less(x,y))
print('For Less_equal: ')
print(np.less_equal(x,y))

###############################################################################

# Write Numpy program to create 3X3 Identity matrix

                            # .identity(size like 3)

import numpy as np
arr1=np.identity(3)
print(arr1)


###############################################################################

# write Numpy program to generate a random number

# .random.normal(loc:mean,scale:standard deviation,size )
                            
import numpy as np
rand_num=np.random.normal(0,1,2)
print(rand_num)

###############################################################################

# write Numpy program to create a
# 3X4 array and iterate over it.

                      # nditer(array)
                      # arrange(start,end) and reshape(row,column)

import numpy as np

a=np.arange(10,22).reshape((3,4))
print('Original Array')
print(a)

for x in np.nditer(a):
    print(x,end=" ")
    print()

###############################################################################

# write Numpy program to create vector of length 5
#a whose values is distributed between 10 and 50
 
              # linspace(start pt,end pt,total no in vector)

import numpy as np
v=np.linspace(10,49,5)
print(v)

###############################################################################
# write Numpy program to create 3X3 matrix 
# with values ranging from 2 to 10

import numpy as np
x=np.arange(2,11).reshape(3,3)
print(x)

###############################################################################

# write Numpy program to reverse an array
# (the first element becomes last)

                                  # [::-1] 

import numpy as np
arr=np.array([1,2,3,4,5,6])
x=arr[::-1]
print(x)

###############################################################################

# write Numpy program to compute the multiplication of two given matric

                                  # dot(matrix1,matrix2)      

import numpy as np
p=[[1,0],[0,1]]
q=[[1,2],[3,4]]
x=np.dot(p, q)
print(x) # multiplication of matrix by .dot

###############################################################################

# write Numpy program to compute cross product of two matrix

import numpy as np
p=[[1,0],[0,1]]
q=[[1,2],[3,4]]
z=np.cross(p, q)
print(z)

###############################################################################

# write Numpy program to compute determinent of given square array

                                    #linalg
                                    #linalg.det(single argument)
                                    #to find determinent

import numpy as np
from numpy import linalg as LA 

p=[[1,0],[1,2]]
q=[[1,2],[3,4]]

x=np.linalg.det(p)
print(x)

###############################################################################

# write Numpy program to compute eigenvalues and right eigenvectors of given square array
  

                                #np.linalg.eig(matrix)

import numpy as np
from numpy import linalg 
a=np.mat("3 -2;1 0")
print('Original Matrix: ',a)
w,v=np.linalg.eig(a)
print('Eigen values: ',w)
print('Eigen Vector: ',v)

###############################################################################

# write Numpy program to compute the inverse of given array

                                 # np.linalg.inv(matrix)
                                 


import numpy as np
from  numpy import linalg
a=np.mat("1,2;3,4")
a=np.linalg.inv(a)
print(a)


###############################################################################

# write Numpy program to compute 




