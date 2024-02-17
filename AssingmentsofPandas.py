# -*- coding: utf-8 -*-
"""
Created on Wed May  3 07:13:17 2023

@author: santo
"""

import pandas as pd
department = ({'Computer':["SE","OSA","CR","M3","ADS","DBMS","xyz"],
               'Project':['ABC','CDE','EFG','GHI','IJK','KLM','MNO'],
              'duration':['25','35','45','55','65','75','85'],
              'fees':['4300','2300','4500','4569','5670','8900','5460']
        })
df = pd.DataFrame(department)
print(df)

#
#
row_label=['r0','r1','r2','r3','r4','r5','r6']
df = pd.DataFrame( department ,index=row_label)
columns=['A','B','C']
df2=df.rename(column={'A':'A1','B':'B1','C':'C1'})
print(df2)
# 
df.shape
#

df.size
#

df.columns
#
df.columns.values
#
df.index
#
df.dtypes
#
df2=df[2:]
df2
#
df['duration'][2]
#
df['fees']=df['fees']-500
df['fees']
df.describe()
#
df=pd.DataFrame(department,index=row_label)
df.columns=['A','B','C','D']
df
#
df=pd.DataFrame(department,index=row_label)
df.columns=['A','B','C','D']
df.dtypes
df2=df.rename({'A':'A1','B':'B1'}, axis=1)
df2=df.rename({'C':'C1','D':'D1'}, axis='columns')
df2=df.rename(columns={'A':'A1','B':'B1'})
print(df2)
#
df2 = df.convert_dtypes()
print(df2.dtypes)
#
df = df.astypes(str)
print(df.dtypes)
#
df = df.astype({"Computer":str,"fees":int,"duration":str})
print(df.dtypes)
#

#
path= r"C:/Users/santo/auto.xlsx.xls"
path2= r"C:/Users/santo/auto.csv.csv"
import pandas as pd
data =pd.read_excel(path)
print(df)
#
df.size
#
df.index
#
df.columns
#
df.shape
#
df.dtypes
#
df2=df[2:]
df2
#
df['State'][2]
#
df2 = df.convert_dtypes()
print(df2.dtypes)
#
cols=['Customer','State']
df[cols]=df[cols].astype('string')
df.dtypes
#
df = df.astype({"Customer":int}, errors='ignore')
df.dtypes
#
df = df.astype({"Customer":int}, errors='raise')
#
df1=df.drop(df.index[1])
df1
#
df1=df.drop(df.index[[1,3]])
df1
#
df1=df.drop(df.index[1:])
df1
#
df1=df.drop([0,3])
df1
#
df.columns=['A','B','C','D']
df.dtypes
df2=df.rename({'Customer':'A1','State':'B1'}, axis=1)
print(df2)
#######################################################################
############################ 9 May ###################################
######################################################################
import pandas as pd
technologies=({'cource':["php","pychamp","cpp","c"],
               'fees':['20000','30000','40000','1200'],
              'duration':['20days','30days','40days','56days'],
        })
index_labels=['r1','r2','r3','r4']

df1 = pd.DataFrame(technologies,index=index_labels)

###########################################################################

import pandas as pd
technologies1=({'cource':["php","pychamp","c","cpp"],

              'discount':[123.2,234.2,456.7,643.5]
})
index_labels1=['r1','r6','r3','r5']
df2 = pd.DataFrame(technologies1,index=index_labels1)
#pandas join , shows left join
df3=df1.join(df2,lsuffix='_left',rsuffix='_right')
print(df3)
#pandas join , shows inner join
df3=df1.join(df2,lsuffix='_left',rsuffix='_right',how='inner')
print(df3)
#pandas join , shows left join
df3=df1.join(df2,lsuffix='_left',rsuffix='_right',how='left')
print(df3)
#pandas join , shows right join
df3=df1.join(df2,lsuffix='_left',rsuffix='_right',how='right')
print(df3)
# Merging DataFrames
import pandas as pd
technologies=({'cource':["php","pychamp","cpp","c"],
               'fees':['20000','30000','40000','1200'],
               })
index_labels=['r7','r8','r3','r9']

df = pd.DataFrame(technologies,index=index_labels)
print(df)
#######################################################################
import pandas as pd
technologies=({'cource':["php","pychamp","cpp","c"],
               'fees':['20000','30000','40000','1200'],
              'duration':['20days','30days','40days','56days'],
        })
index_labels=['r1','r2','r3','r4']

df1 = pd.DataFrame(technologies,index=index_labels)
#####################################################################
import pandas as pd
technologies1=({'cource':["php","pychamp","c","cpp"],

              'discount':[123.2,234.2,456.7,643.5]
})
index_labels1=['r1','r6','r3','r5']
df2 = pd.DataFrame(technologies1,index=index_labels1)
#####################################################################
#merging two dataframe
df3=pd.merge(df1,df2)
df3
#######################################################################
df3=df1.merge(df2)
df3
# concat() mtd use for adding DataFrame Horizontaly ################
data=[df1,df2]
df2=pd.concat(data)
df2
#
df3=pd.concat([df,df1,df2])
df3
#
df.to_csv("C:/Users/santo/data_file.csv")
print(df)
###########################################################################
##pandas join on column left

df3=df1.set_index('cource').join(df2.set_index('cource'),how='inner')
print(df3)

#pandas join on column left
df3=df1.set_index('cource').join(df2.set_index('cource'),how='left')
print(df3)

##pandas join on column right
df4=df1.set_index('cource').join(df2.set_index('cource'),how='right')
print(df4)
######################################################################
###########SERIES 9 May################################################
######################################################################
import pandas as pd
songs2=pd.Series([145,142,38,13],name='counts')
songs2.index
#################################################################
songs3 = pd.Series([145,142,38,13],name='counts',
    index=['Argit','Yo_yo','Bdshah','Guru'])
songs3.index
songs3
# finding mean
import numpy as np
numpy_ser=np.array([145,142,38,13])
songs3[1]
songs3.mean()
# CURD Operation on Series 
#CREate
sita=pd.Series([12,23,45,67],
index=['xyz','abc','jkl','mno'],
name='sita class')
sita
#Read
sita['abc']
sita['xyz']
for item in sita:
    print(item)
#Update
sita['abc']=89
sita['abc']
sita
# Delete
s=pd.Series[1,2,3,4],index=[1,2,5,7]
del s[1]
#Convert Datatype
songs_66=pd.Series([3,None,11,9],
index=['ABC','XYZ','JKL','MNO'],
name='counts')
pd.to_numeric(songs_66.apply(str))
pd.to_numeric(songs_66.astype(str),errors='coerce')
#
songs_66.fillna(-1)
#
songs_66.dropna()
####Append
import pandas as pd
songs_69=pd.Series([7,21,14,9],
index=['ABC','XYZ','JKL','MNO'],
name='counts')
songs=songs_69.append(songs_69)
##### ploting point graph
import matplotlib.pyplot as plt
fig=plt.figure
songs_69.plot()
plt.legend()
###### ploting bar graph
fig=plt.figure()
songs_69.plot(kind='bar')
songs_69.plot(kind='bar',color='k',alpha=5)
plt.legend()
# HISTOGRAM
import pandas as pd
import numpy as np
data=pd.Series(np.random.randn(500),
name='500 random')
fig=plt.figure()
ax=fig.add_subplot(111)
data.hist()
# creat np array
import numpy as np
arr1=np.array([7,20,13])
arr2=np.array([3,5,2])
arr1
arr1.dtype
# multidimentional array
arr=np.array([[7,20,13],[1,2,3]])
print(arr)
# ndmin
arr=np.array([23,45,67,89],ndmin=2)
print(arr)
#
arr=np.array([23,45,67,89],ndmin=7)
print(arr)
# change datatype into complex
arr=np.array([23,45,67,89],dtype=complex)
print(arr)
# change datatype into float
arr=np.array([23,45,67,89],dtype=float)
print(arr)
# get dimension of array
arr=np.array([[7,20,13],[67,54,67],[1,2,3]])
print(arr.ndim)
print(arr)
# searching bytes size of each item (int contain 4 bytes)
arr=np.array([[7,20,13],[67,54,67],[1,2,3]])
print('each item contain in bytes:', arr.itemsize)
#
arr=np.array([[7,20,13],[67,54,67],[1,2,3]])
print('each item contain in which dytes:', arr.dtype)
#size
arr=np.array([[7,20,13],[67,54,67],[1,2,3]])
arr.size
# shape 
arr.shape
# in the form of list
arr=np.array([[7,20,13],[67,54,67],[1,2,3]])
print("Array created by using list: /n" ,arr)
# in the form of list with dtype=float
arr=np.array([[7,20,13],[67,54,67],[1,2,3]] ,dtype='float')
print("Array created by using list: /n" ,arr)
# difine in range 
arr=np.arange(0,20,3)
print("A squential array with stepe of 3:/n",arr)
# rangeof sequential array
import numpy as np
arr=np.arrange(11)
print(arr)
#array element index no
print(arr[2])
##array element index no
print(arr[-2])
# accesing element in multidimentional array
arr=np.array([[7,20,13,67,54],[67,66,17,2,3]])
print(arr)
#
print(arr.size)
#
print(arr.shape)
#
print(arr[1,1])
#
print(arr[0,4])
# accesing element by slicing
arr=np.array([0,1,2,3,4,5,6,7,8,9])
x=arr[1:8:2]
print(x)
#
x=arr[-2:3:-1]
print(x)
#
x=arr[-2:10]
print(x)
#
multi_arr=np.array([[10,20,10,40],
            [40,50,70,90],
            [60,10,70,80],
            [30,90,40,30]])
#
multi_arr[1,2]
#
multi_arr[:,1]
#
multi_arr[1,:]
#
x=arr[:3,::2]
print(x)
#integer array indexing
import numpy as np
arr=np.arange(35).reshape(5,7)
print(arr)


############# 11may ################################################
import numpy as np
arr=np.arange(12).reshape(3,4)
print(arr)
#Boolean Array Indexing
rows=np.array([False,True,True])
wanted_rows=arr[rows,:]
print(wanted_rows)

# tolist #############################################################
#array to np
array=np.array([10,20,30,40])
print("Array:", array)
print(type(array))

#array to list (single dimensional)####################################
list=array.tolist()
print("List:", list)
print(type(list))

#multidimansional array to list #######################################
lst=array.tolist()
print("List:", list)

# as array (np.asarray) ##############################################
lst=[10,20,30,40]
array=np.asarray([10,20,30,40])
print("Array:", array)
print(type(array))

# Numpy Array Properties #############################################

#ndarray.shape ######################################################
array=np.asarray([[1,2,3],[4,5,6]])
print(array.shape)

# array =(3,2) size # resize mtd #####################################
array=np.asarray([[1,2,3],[4,5,6]])
array.shape=(3,2)
print(array)

#ARITHMATIC OPERATION ON ARRAY ########################################################
# array reshape mtd ###################################################
array=np.asarray([[1,2,3],[4,5,6]])
new_array=array.reshape(3,2)
print(new_array)

# Anather reshape (matrix) #############################################
arr1=np.array(16).reshape(4,4)
arr2=np.array([1,2,3,4])

# Addition ############################################################
add_arr=np.add(arr1,arr2)
print(f"Adding two array :\n{add_arr}")

#Subtraction ##########################################################
sub_arr=np.subtract(arr1,arr2)
print(f"Sub two array :\n{sub_arr}")
#
sub_arr=np.subtract(arr1,arr2)
sub_arr

#multiply ###############################################################
mul_arr=np.multiply(arr1,arr2)
print(f"Adding two array :\n{mul_arr}")
#
mul_arr=np.multiply(arr1,arr2)
mul_arr

# divide ################################################################
div_arr=np.divide(arr1,arr2)
print(f"division two array :\n{div_arr}")

#
div_arr=np.divide(arr1,arr2)
div_arr

#Reciprocal  ######################################################## 
# Reduction ######################################################### 
arr1=np.array([50,10.3,5,1,200])
rep_arr1=np.reciprocal(arr1)
rep_arr1

# to inhance the image Numpy Power ########################################################
arr1=np.array([3,10,5])
pow_arr1=np.power(arr1,3)
pow_arr1

#
arr2=np.array([3,2,1])
pow_arr2=np.power(arr1,arr2)
print(arr2)

################################################################################################################
################################################################################################################

#ndarray.size ########################################################
array=np.asarray([[1,2,3],[4,5,6]])
print(array.size)

#ndarray.dtype ########################################################
array=np.asarray([[1,2,3],[4,5,6]])
print(array.dtype)

# ndarray.item ########################################################
array=np.asarray([[1,2,3],
                  [4,5,6]])
print(array.item)

#ndarray.ndim for finding dimension ########################################################
#array dimention can define by no of bracket  ########################################################
# no of array= no of bracket) ########################################################
# color array define by ............. ########################################################
array=np.asarray([[1,2,3],[4,5,6]])
array.ndim

#ndarray.itemsize ########################################################
array=np.asarray([[1,2,3],[4,5,6]])
print(array.size)

# 
import pandas as pd
arr1=np.array([7,20,13])
arr1=np.array([3,5,2])
arr1
arr1.dtype
mod_arr=np.mod(arr1,arr2)
mod_arr

# CREATE EMPTY ARRAY ########################################################
from numpy import empty
a= empty([3,3])
print(a)

#CREATE ZERO ARRAY ########################################################
from numpy import zeros
a=zeros([3,5])
print(a)

# CREATE ONE ARRAY #####################################################33
from numpy import ones
a=ones([5])
print(a)

# CREATE ARRAY WITH VSTACK ##########################################
from numpy import array
from numpy import vstack
#CREATE FIRST ARRAY
a1=array([1,2,3])
print(a1)

#CREATE SECOND ARRAY ####################################################
a2=array([4,5,6])
print(a2)

#CREATE THIRD ARRAY VERTICAL STACK ############################################
a3=vstack((a1,a2))
print(a3)
print(a3.shape)

#CREATE ARRAY WITH HSTACK ###########################################
from numpy import array
from numpy import hstack

#CREATE FIRST ARRAY ##################################################
a1=array([1,2,3])
print(a1)

#CREATE SECOND ARRAY ####################################################
a2=array([4,5,6])
print(a2)

#CREATE THIRD ARRAY VERTICAL STACK ############################################
a3=hstack((a1,a2))
print(a3)
print(a3.shape)

# INDEX ARRAY INDEX OUT OF BOUNDS ######################################
import numpy as array
data=([11,22,33,44,55])
print(data[5])

#
from numpy import array
data=([11,22,33,44,55])
import numpy as array
print(data[-1])
print(data[-5])

#index row of one dimensional array
from numpy import array
data=array([
   [11,22],
   [33,44],
   [55,66]])
print(data[0,])

############ 12MAY ##################################################
#####################################################################

#index row of two dimensional array
from numpy import array
data=array([
   [11,22],
   [33,44],
   [55,66]])
print(data[0,0])

#negative slicing of one dimensional array ##########################
from numpy import array
data=array([11,22,33,44,55,66])
print(data[-2:])

#
from numpy import array
data=array([
   [11,22,33],
   [44,55,66],
   [77,88,99]])
x ,y =data[:,:-1], data[:,-1]

# broadcast scalar to one dimensional array ############################
from numpy import array
a=array([1,2,3])
print(a)
#
b=2
print(b)
#
c=3
print(c)
# add vector
c=a+b
print(c)
######################################################################
# vector L1 norm
from numpy import array
from numpy.linalg import norm
a=array([1,2,3])
print(a)
11 == norm(a,1)
print(11)

# vector L2 norm
from numpy import array
from numpy.linalg import norm
a=array([1,2,3])
print(a)
12 == norm(a)
print(12)

# triangular Matrics
from numpy import array
from numpy import tril
from numpy import triu
M = array([
    [1,2,3],
    [1,2,3],
    [1,2,3]])
print(M)
# lower triangular
lower=tril(M)
print(lower)
# upper triangular
upper=triu(M)
print(upper)

#diagonal matrix
from numpy import array
from numpy import diag
M = array([
    [1,2,3],
    [1,2,3],
    [1,2,3]])
print(M)
# extract digonal vector
d=diag(M)
print(d)
# create digonal matrix from vector
D=diag(M)
print(D)

#identity matrics
from numpy import identity
I = identity(3)
print(I) 

#orthogonal matrix
from numpy import array
from numpy.linalg import inv
Q = array([
    [1,0],
    [0,-1]])
print(Q)

# inverse Equivalence
V=inv(Q)
print(Q.T)
print(V)

# identify equivalance
I =Q.dot(Q.T)
print(I)

# single line plot
import matplotlib.pyplot as plt
plt.plot([1,3,2,4])
plt.show()

#multiline plot###################################################
import matplotlib.pyplot as plt
x=range(1,5)
# mathematical  list comprehension 
plt.plot(x,[xi*1.5 for xi in x])
plt.plot(x,[xi*3.0 for xi in x])
plt.plot(x,[xi/3.0 for xi in x])
plt.show()

#grid
import matplotlib.pyplot as plt
x=range(1,5)
# mathematical  list comprehension 
plt.plot(x,[xi*1.5 for xi in x])
plt.plot(x,[xi*3.0 for xi in x])
plt.plot(x,[xi/3.0 for xi in x])
plt.grid(True)
plt.show()

#
import matplotlib.pyplot as plt
import numpy as np
x = np.arange (1,5)
plt.plot(x,x*1.5, x, x*3.0, x, x/3.0)
plt.grid(True)
plt.show()

#axis
import matplotlib.pyplot as plt
import numpy as np
x = np.arange (1,5)
plt.plot(x,x*1.5, x, x*3.0, x, x/3.0)
plt.axis()#show the current axis lmt values

#new axis limit
plt.axis([0,5,-1,13])
plt.show()

#label
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.xlabel("THIS IS X Axis")
plt.ylabel("THIS IS Y Axis")
plt.show()

# Title
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.title('Sample')
plt.show()

#legend
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5)
plt.plot(x,x*1.5, label="normal")
plt.plot(x,x*3.0, label="fast")
plt.plot(x,x/3.0, label="slow")
plt.legend()
plt.show()

# Control Color
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3)
plt.plot(y,'y')
plt.plot(y+1,'m')
plt.plot(y+2,'c')
plt.show()

#specifing style in multiline graph
import matplotlib.pyplot as plt
y=np.arange(1,3)
plt.plot(y,'y',y+1,'m',y+2,'c');
plt.show()

#control line  style
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3)
plt.plot(y,'--',y+1,'--',y+2,':');
plt.show()

# Markers style
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3,0.2)
plt.plot(y,'x',y+0.5,'o',y+1.5,'^',y+2,'s');
plt.show()

#histogram 
import matplotlib.pyplot as plt
import numpy as np
y=np.random.randn(1000)
plt.hist(y)
plt.show()

#bar used to generate bar chart in matplotlab
#it is univeriable
#the function expects two lists of values:
# the x coordinaters that are the positions 
# of the bar's left
#[1,2,3] square bracket defines  x axis 
#[3,2,5] square bracket defines height
import matplotlib.pyplot as plt
plt.bar([1,2,3],[3,2,5]);
plt.show()

#scatter plots
# it displays two set of data
# imp in regration analysis
# x and y must be same size
import matplotlib.pyplot as plt
import numpy as np
x=np.random.randn(100)
y=np.random.randn(100)
plt.scatter(x,y);
plt.show()
    
# changing color
size=20*np.random.randn(100)
colors =np.random.randn(100)
plt.scatter(x,y, s=size, c=colors);
plt.show()

# adding text
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-4,4,1024)
y=.25*(x+4.)*(x+1)*(x-2.)
plt.text(-0.5, -0.25, 'Brak min')
plt.plot(x,y,c='k')
plt.show()

#seaborn
import seaborn as sns
import pandas 
import matplotlib.pyplot as plt
sns.get_dataset_names()
df = sns.load_dataset('titanic')
df.head()

# 13May #####################################################
# countplot for seaborn quality graph 
sns.countplot(x='sex',data=df)

# hue concept
sns.countplot(x='sex',hue='survived',data=df,palette='Set1')
sns.countplot(x='sex',hue='survived',data=df,palette='Set2')
sns.countplot(x='sex',hue='survived',data=df,palette='Set3')

#hue=the name of the catagorical column to split the graph
#kde plot=kernal density estimate for distribution
#of data
sns.kdeplot(x='age', data=df,color='black')

#bins shows square graph shows no of bars=bins
sns.displot(x='age',kde='True',bins=6,data=df)

#
sns.displot(x='age',kde='True',bins=5,
hue=df['survived'],palette='Set1',data=df)

#dataset of 'iris'
import seaborn as sns
import pandas 
import matplotlib.pyplot as plt
sns.get_dataset_names()
df = sns.load_dataset('iris')
df.head()


# graph related speed will display by given with displot
sns.displot(x='sepal_length',kde=True,bins=6,data=df) 
# graph related distance will display by given with displot
sns.displot(x='petal_length',kde=True,bins=6,data=df)

#boxplot
df.describe()
sns.boxplot(df.sepal_length)

# histogram of distance
plt.hist(df.sepal_length)

#boxplot on dist
sns.boxplot(df.sepal_length)

#skeweness of speed
from scipy.stats import skew
from scipy.stats import kurtosis
sepal_length =df['sepal_length'].tolist()
sepal_length
print("skewnes of sepal_length ", skew(sepal_length))
dist=df['sepal_length'].tolist()
print("skewness of sepal_length", skew(sepal_length))
print(skew( sepal_length, axis=0,bias=True))
#
sns.boxplot(df.sepal_length)

#
sns.countplot(x='sepal_length',data=df)

#
sns.countplot(x='sepal_length',data=df,palette='Set1')

#
#scatterplot
#it helps to understand corelation between data and lenghts of flower
sns.scatterplot(x='sepal_length',y='petal_length',data=df,hue='species')

# regular jointplot graph 
sns.jointplot(x='sepal_length',y='petal_length',data=df,kind='reg') 
# histogram for hointplot
sns.jointplot(x='sepal_length',y='petal_length',data=df,kind='hist')
# kde graph of jointplot
sns.jointplot(x='sepal_length',y='petal_length',data=df,kind='kde')


#relationship of features by pair plot
#pair plot 
sns.pairplot(df)

# heatmap
#corr= corelation
corr = df.corr()
sns.heatmap(corr)

#
sns.boxplot(df)

#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
cars=pd.read_csv("C:/Users/santo/cars.csv.csv")
cars.columns # about columns
cars.describe() #discribes the DataFrame

# histogram of cars speed
plt.hist(cars.speed)
# graph related speed will display by given with displot
sns.displot(x='speed',kde=True,bins=6,data=cars) 
# graph related distance will display by given with displot
sns.displot(x='dist',kde=True,bins=6,data=cars)

#boxplot
cars.describe()
sns.boxplot(cars.speed)

# histogram of distance
plt.hist(cars.dist)

#boxplot on dist
sns.boxplot(cars.dist)

#skeweness of speed
from scipy.stats import skew
from scipy.stats import kurtosis
speed =cars['speed'].tolist()
speed
print("skewnes of speed ", skew(speed))
dist=cars['dist'].tolist()
print("skewnes of dist ", skew(dist))
print(skew(dist, axis=0,bias=True))
#
sns.boxplot(cars.speed)
#
sns.pairplot(cars)










































































































































































