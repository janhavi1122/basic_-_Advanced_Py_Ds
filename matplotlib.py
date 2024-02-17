# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:18:03 2023

@author: hp
"""

'''MATPLOTLIB'''
#Important for virsulization
#it can understand data ,preprocessos  data

import matplotlib.pyplot as plt
plt.plot([1,3,2,4])
plt.show()

#########################################
#multiline plots
import matplotlib.pyplot as plt
x=range(1,5)
plt.plot(x,[xi*1.5 for xi in x])

plt.plot(x,[xi*3.0 for xi in x])

plt.plot(x,[xi/3.0 for xi in x])
plt.show()
###########################################
#how matplotlib automatically
#choose different color

import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5)
plt.plot(x,x*1.5,x,x*3.0,x,x/3.0)
plt.grid(True)
plt.show()
#######################################
#handling the axes
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1 ,5)
plt.plot(x,x*1.5,x,x*3.0,x,x/3.0)
plt.axis()
plt.axis([0,5,-1,13])
plt.show()
###################################################
#adding the label
import matplotlib.pyplot as plt
plt.plot([1,3,2,4])
plt.xlabel("This is the x-axis")
plt.ylabel("This is the y-axis")
plt.show()
##########################################################
#adding title

import matplotlib.pyplot as plt
plt.plot([1,3,2,4])
plt.title('simple plot')
plt.show()
#matplotlib provide a simple function,plt.title()
############################################
#adding legend(give the detail about that graph)

import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5)
plt.plot(x,x*1.5,label='normal')
plt.plot(x,x*3.0,label='fast')
plt.plot(x,x/3.0,label='slow')
plt.legend()
plt.show()
#############################################
'''color abbreviation
color name
b  blue
c cyan
g green
k black
m magenta
r red
w white
y yellow
'''
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3)
plt.plot(y,'y');
plt.plot(y+1,'m');
plt.plot(y+2,'c');
plt.show()
######################################
#control line style
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3)
plt.plot(y,'--',y+1,'-*',y+2,':');
plt.show
################################################
''' style abbreviation style
solid line
dashed line
dash-dot
'''
'''
marker
+ plus marker
X cross marker
D diamond marker

'''
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3,0.2)
plt.plot(y,'x',y+0.5,'o',y+1,'D',y+1.5,'^',y+2,'s')
plt.show()

###################################################
#Histogram chart
import matplotlib.pyplot as plt
import numpy as np
y=np.random.randn(1000)
plt.hist(y);#imp *************
plt.show()

#######################################
#Bar graph
import matplotlib.pyplot as plt
plt.bar([1,2,3],[3,2,11])
plt.show()
'''
bar function used to generate bar graph
'''
############################################
#scatter plot
'''it display the value of '''
import matplotlib.pyplot as plt
import numpy as np
x=np.random.randn(1000)
y=np.random.randn(1000)
plt.scatter(x,y);
plt.show()

########################################
size=50*np.random.randn(1000)
colors=np.random.rand(1000)
plt.scatter(x,y,s=size,c=colors);
plt.show()

########################################
#Adding text
import numpy as np
import matplotlib.pyplot as plt
X=np.linspace(-4,4,1024)
Y=.25 * (X+4.)*(X + 1.)*(X-2.)
plt.text(-0.5,-0.25,'Brackmard minimum')
plt.plot(X,Y,c='k')
plt.show()

###################################################
#how to use seaborn for data virtualization
#pip install seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#seaborn is 18 in-build dataset
#that can be found using the following command
sns.get_dataset_names()
df=sns.load_dataset('titanic')
df.head()

#############################################
#count plot
'''when dealing with categorical value
it is used to plot the frequency
of the different categories
'''
sns.countplot(x='sex',data=df)
#x -the name of column
#data-the dataframe

sns.countplot(x='sex',hue='survived',data=df,palette='Set1')
sns.countplot(x='sex',hue='survived',data=df,palette='Set2')
sns.countplot(x='sex',hue='survived',data=df,palette='Set3')
#hue-the name of categorical column to split the bar
#plateee-the color of paletee to be used
#KDE PLOT
#A kernal density estimate plot are used
################################################
#to plot the distribution of continuose data

sns.kdeplot(x='age',data=df,color='black')

##############################################
#x-the name of the column
#data -the dataframe
#color-color of the graph
sns.displot(x='age',kde=True,bins=6,data=df)

######################################################
sns.displot(x='age',kde=True,bins=5,
     hue=df['survived'],palette='Set1',data=df)

##################################################
df=sns.load_dataset('iris')
df.head()

#################################################
#scatter plot help understand corelation between data
sns.scatterplot(x='sepal_length',y='petal_length',
                data=df,hue='species')
##################################################

#join plot
#a join data is corelation 
sns.jointplot(x='sepal_length',y='petal_length',
data=df,kind='reg')

sns.jointplot(x='sepal_length',y='petal_length',
data=df,kind='hist')

sns.jointplot(x='sepal_length',y='petal_length',
data=df,kind='kde')

###################################
#pair plot
sns.pairplot(df)
#################################
#heat map can be used to visulize confusion

corr=df.corr()
sns.heatmap(corr)
####################################