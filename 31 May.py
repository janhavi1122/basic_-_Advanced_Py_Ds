# -*- coding: utf-8 -*-
"""
Created on Wed May 31 03:47:10 2023

@author: santo
"""
import pandas as pd
path=r"C:\Users\santo\Downloads\Telegram Desktop\Diabetes.csv"
df = pd.read_csv(path)
print(df)

#length of row #######################################################
rows_count=len(df.index)
rows_count

#length of row by axes################################################
rows_count=len(df.axes[0])
rows_count

#count of row occupied ##############################################
rows_count=df.shape[0]
rows_count=df.shape[1]
print(rows_count)

#get the list of all column names from headers#######################
column_headers = list(df.columns.values)
print("column_headers:",column_headers)

#list (df) to get column headers as list##############################
column_headers=list(df.columns)
column_headers

#list column by using df ############################################
column_headers=list(df)
column_headers

#pandas shuffle DataFrame on rows #####################################
# for simple suffle program ##########################################
df1=df.sample(frac=1)
print(df1)

#for simple suffle program half data ###################################
df1=df.sample(frac=0.5)
print(df1)

#for reseting shuffle from 0th row ###################################
df1=df.sample(frac=1).reset_index()
print(df1)

# showing only original sequence from 0 ################################
df1=df.sample(frac=1).reset_index(drop=True)
print(df1)



import pandas as pd
path=r"C:\Users\santo\Downloads\Telegram Desktop\ethnic diversity.csv"
df = pd.read_csv(path)
print(df)

#length of row #######################################################
rows_count=len(df.index)
rows_count

#length of row by axes################################################
rows_count=len(df.axes[0])
rows_count

#count of row occupied ##############################################
rows_count=df.shape[0]
rows_count=df.shape[1]
print(rows_count)

#get the list of all column names from headers#######################
column_headers = list(df.columns.values)
print("column_headers:",column_headers)

#list (df) to get column headers as list##############################
column_headers=list(df.columns)
column_headers

#list column by using df ############################################
column_headers=list(df)
column_headers

#pandas shuffle DataFrame on rows #####################################
# for simple suffle program ##########################################
df1=df.sample(frac=1)
print(df1)

#for simple suffle program half data ###################################
df1=df.sample(frac=0.5)
print(df1)

#for reseting shuffle from 0th row ###################################
df1=df.sample(frac=1).reset_index()
print(df1)

# showing only original sequence from 0 ################################
df1=df.sample(frac=1).reset_index(drop=True)
print(df1)
#31 may
import numpy as np
print(np.__version__)
print(np.show_config())

print(np.info(np.add))

#to test nune of the array zero
x=np.array([1,2,3,4])
print("original array:")
print(x)
print("to test nune of the array zero")
print(np.all(x))

#
x=np.array([0,1,2,3])
print("original array:")
print(x)
print("to test nune of the array zero")
print(np.all(x))
# for non-zero
x=np.array([1,0,0,0])
print(x)
print("to test nune of the array non-zero")
print(np.any(x))
#
a=np.array([1,0,np.nan])
print("original array:")
print(x)
print("to test nune of the array infinite")
print(np.infinite(x))

#1st June
#cross product
import numpy as np
p=[[1,0],[0,1]]
q=[[1,2],[2,1]]
print("original matrix:")
print(p)
print(q)
result1=np.cross(p,q)
result2=np.cross(q,p)
print("cross product of (p,q)")
print(result1)
print("cross product of (q,p)")
print(result2)
#determinant
import numpy as np
from numpy import linalg as LA
a=np.array([[1,0],[1,2]])
print("Original 2-d array")
print(a)
print("Determinant")
print(np.linalg.det(a))
#eigenvalue and eigenvector
import numpy as np
m=np.mat("3 -2;1 0")
print("Original matrix")
print("a\n",m)
w,v =np.linalg.eig(m)
print("Eigenvector of ",w)
print("Eigenvalues of ",v)
#inverse of matrix
import numpy as np
m=np.array([[1,2],[3,4]])
print("Original 2-d array")
print(m)
result=np.linalg.inv(m)
print("inverse of given matrix:")
print(result)
#normal distribution
import numpy as np
x=np.random.normal(size=5)
print(x)
#random no 6 between 10 to 30
import numpy as np
x=np.random.randint(low=10,high=30,size=6)
print(x)
#3*3*3*
import numpy as np
x=np.random.random(3,3,3)
print(x)
#5by 5 array and random value to find min and max valaue
import numpy as np
x=np.random.random(5,5)
print("Original 2-d array")
print(x)
xmin,xmax =x.min(),x.max()
print("Min and Max values:")
print(xmin,xmax)

#2nd June
import numpy as np
x=np.arange(4).reshape((2,2))
print("\nOriginal array:")
print(x)
print("\nMax values along second axis:")
print(np.amax(x,1))
print("\nMin values along second axis:")
print(np.amin(x,1))
#line with suitable label
import matplotlib.pyplot as plt
X=range(1,50)
Y=[value *3 for value in X]
print("Values of X:")
print(*range(1,50))
print("Vlaues of Y (thrice of X):")
print(Y)
plt.plot(X,Y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title('Draw a line')
plt.show()
#line with suitable label with values
import matplotlib.pyplot as plt
X=[1,2,3]
Y=[2,4,1]
plt.plot(X,Y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title('A simple Graph')
plt.show()
#graph of csv file
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("C:/Users/santo/fdata.csv")
df.plot()
df.show()

import matplotlib.pyplot as plt
import pandas as pd
att=pd.read_csv("C:/Users/santo/Downloads/Telegram Desktop/Data_Science_Attendance_Sheet2.csv")
att
pos=list(df['datum'])
height=list(df['Parjane Pranjal'])
height.remove(103)
pos.remove(np.nan)
pos
height
df2=df.drop('year',axis=1)
df2=df2.drop('month',axis=1)
df2=df2.drop('weekday',axis=1)
df2=df2.drop('datum',axis=1)
df2=df2.drop[:-1]
df2.plot()

#
import matplotlib.pyplot as plt
x1=[10,20,30]
y1=[20,40,10]
x2=[10,20,30]
y2=[20,10,30]
plt.plot(X,Y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title('A simple Graph')
plt.plot(x1,y1,color='blue',linewidth =3,label='line-width-3')
plt.plot(x2,y2,color='red',linewidth =5,label='line-width-5')
df.legend()
df.show()