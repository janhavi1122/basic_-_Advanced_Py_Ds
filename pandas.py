# -*- coding: utf-8 -*-
"""
Created on Mon May  1 20:48:21 2023

@author: santo
"""

import pandas as pd
pd.__version____
#
import pandas as pd
technologies =  [["spark",2000,"30days"],
                 ["pandas",2000,"40days"]
                ]
df = pd.DataFrame(technologies)
print(df)
#
columns_name=["cource","fees","duration"]
row_label=["a","b"]
df=pd.DataFrame(technologies,columns=columns_name,index=row_label)
print(df)
#
df.dtypes
#
types={'cource':int,'fees':float,'duration':str}
df.dtypes
#
technologies={'cource':["php","py","c++"],
              'fees':['20000','30000','40000'],
              'duration':['20days','30days','40days']
              }
df=pd.DataFrame(technologies)
df
#
df.to_csv('C:/Users/santo/data_file.csv')
#
df = pd.read_csv('C:/Users/santo/data_file.csv')
#
import pandas as pd

technologies=({'cource':["php","py","c++"],
               'fees':['20000','30000','40000'],
              'duration':['20days','30days','40days']
        })
row_label=['r0','r1','r2']
df = pd.DataFrame(technologies,index=row_label)
print (df)
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
df['fees']
#
df[['cource','fees']]
#
df2=df[2:]
df2
#
df3=df[2:]
df3
#
df['duration'][2]
#
df['fees']=df['fees']-500
df['fees']
df.describe()
#
df=pd.DataFrame(technologies,index=row_label)
df.columns=['A','B','C']
df
#
df=pd.DataFrame(technologies,index=row_label)
df.columns=['A','B','C']
df.dtypes
df2 = df.rename({'A':'A1','B':'B1'}, axis=1)
df2 = df.rename({'C':'C1'}, axis='columns')
df2 = df.rename(columns={'A':'A1','B':'B1','C':'C1'})
print(df2)
#
import pandas as pd
import numpy as np
technologies=({'cource':["php","py","c++","c","datascience"],
               'fees':['20000','30000','40000','1200','23000'],
              'duration':['20days','30days','40days','56days','34days'],
              'discount':[123.2,234.2,456.7,643.5,453.6]
        })
df = pd.DataFrame(technologies)
print(df)
print (df.dtypes)
df2 = df.convert_dtypes()
print(df2.dtypes)
#astypes  makes all data type same. 
df = df.astypes(str)
print(df.dtypes)
#
df = df.astype({"cource":str,"fees":int,"duration":str})
print(df.dtypes)
#
df=pd.DataFrame(technologies)
df.dtypes
cols=['fees','discount']
df[cols]=df[cols].astype('float')
df.dtypes
#
df=pd.DataFrame(technologies)
df.dtypes
cols=['cource','duration']
df[cols]=df[cols].astype('str')
df.dtypes
#
df = df.astype({"cource":int}, errors='ignore')
df.dtypes
#
df = df.astype({"cource":int}, errors='raise')
#
df=df.astype(str)
print (df.dtypes)
df['discount']=pd.to_numeric(df['discount'])
df.dtypes
#doping rows and columns
row_label=['r0','r1','r2','r3','r4','r5','r6']
df=pd.DataFrame(technologies,index=row_label)
df.columns=['A','B','C','D']
print(df)
#
df1=df.drop(['r1','r2'])
df1
#
df1=df.drop(df.index[1])
df1
df1=df.drop(df.index[[1,3]])
df1
#
df1=df.drop(df.index[1:])
df1
#
df=pd.DataFrame(technologies)
df1=df.drop(0)
df1
#
df=pd.DataFrame(technologies)#it will delete 0 and 1
#
df2=df.drop(["fees"],axis=1)
print(df2)
#
df2=df.drop(labels=['fees'],axis=1)
print(df2)
#
df2=df.drop(columns=['fees'],axis=1)
print(df2)
#drop columns by index#
print(df.drop(df.columns[[1]],axis=1))
df=pd.DataFrame(technologies)
#
df.drop(df.columns[[2]],axis=1,inplace=True)
print(df)
#
df=pd.DataFrame(technologies)
df2=df.drop(df.columns[[0,1]],axis=1)
print(df2)
#
df=pd.DataFrame(technologies)
liscol=["cource","fees"]
df2=df.drop(liscol,axis=1)
print(df2)
#
df.drop(df.columns[1],axis=1,inplace=True)
print(df)
#
df=pd.DataFrame(technologies,index=row_label)
df2=df.iloc[:,0:2]
df2
#
df2=df.iloc[0:2,:]
df2
#
df3=df.iloc[1:2,1:3]
df3
#
df3=df.iloc[:,1:3]
df3
#
df3=df.iloc[2]
df3
#
df2=df.iloc[[2,3,6]]
df2
#
df2=df.iloc[1:5]
df2
#
df2=df.iloc[:1]
df2
#
df2=df.iloc[:3]
df2
#
df2=df.iloc[-1:]
df2
#
df2=df.iloc[::2]
df2
#
row_label=['r0','r1','r2','r3','r4','r5','r6']
df=pd.DataFrame(technologies,index=row_label)
df.columns=['A','B','C','D']
print(df)
df2=df.loc['r1']
df2
#
df2=df.loc[['r2','r3','r4']]
df2
#
df2=df.loc['r1':'r5']
df2
#
df2=df.loc['r1':'r5':2]
df2
#
df2=df['fees']
df2
# select multiple columns
df2=df[["cource","fees","duration"]]
df2
#using loc select multiple columns
df2=df.loc[:,["cource","fees","duration"]]
df2
#select columns between two column
df2=df.loc[:,'fees','duration']
df2
#select column in range
df2=df.loc[:,'duration':]
df2
#every alternative
df2=df.loc[:,:'duration']
df2
#
df2=df.query("cource=='sap'")
print(df2)
##################################################################
##################################################################
path=r"C:\Users\santo\boston_data.csv"
import pandas as pd
df = pd.read_csv(path)
df2=df.drop(["crim"],axis=1)
print(df2)
####################################################################
#DROPING
####################################################################
liscol=["crim","indus"]
df2=df.drop(liscol,axis=1)
print(df2)
#
df2=df.drop(labels=['crim'],axis=1)
print(df2)
#
df2=df.drop(columns=['crim'],axis=1)
print(df2)
#drop columns by index#
print(df.drop(df.columns[[9]],axis=1))
#
df.drop(df.columns[[9]],axis=1,inplace=True)
print(df)
#
df2=df.drop(df.columns[[0,1]],axis=1)
print(df2)
#
liscol=["crim","indus"]
df2=df.drop(liscol,axis=1)
print(df2)
#
df.drop(df.columns[1],axis=1,inplace=True)
print(df)
#####################################################################
#ILOC
#####################################################################
df2=df.iloc[:,0:2]
df2
#
df2=df.iloc[0:2,:]
df2
#
df3=df.iloc[1:2,1:3]
df2
#
df3=df.iloc[:,1:3]
df2
#
df3=df.iloc[2]
df2
#
df2=df.iloc[[2,3,6]]
df2
#
df2=df.iloc[1:5]
df2
#
df2=df.iloc[:1]
df2
#
df2=df.iloc[:3]
df2
#
df2=df.iloc[-1:]
df2
#
df2=df.iloc[::2]
df2
#####################################################################
#LOC
#####################################################################
row_label=['r0','r1','r2','r3','r4','r5','r6']
df.columns=['A','B','C','D']
print(df)
df2=df.loc['r1']
df2
#
df2=df.loc[['r2','r3','r4']]
df2
#
df2=df.loc['r1':'r5']
df2
#
df2=df.loc['r1':'r5':2]
df2
#
df2=df['crim']
df2
# select multiple columns
df2=df[["crim","indus","zn"]]
df2
#using loc select multiple columns
df2=df.loc[:,["crim","indus","zn"]]
df2
#select columns between two column
df2=df.loc[:,'indus','zn']
df2
#select column in range
df2=df.loc[:,'zn':]
df2
#every alternative
df2=df.loc[:,:'zn']
df2
#####################################################################
import pandas as pd
tutors = ['Ram','Sham','Ghansham','Ganesh','Ramesh']
df = pd.DataFrame(tutors)
df2=df.assign(TutorAssignes=tutors)
print(df2)
# Add multiple column 
MNCCompanies =['TATA','BAJAJ','MAHINDRA','GOOGLE','AMEZON']
df2=df.assign(MNCComp= MNCCompanies,TutorAssigned=tutors)
df2
#
lst=[MNCCompanies.capitalize() for MNCCompanies in MNCCompanies]   
print(lst)
# Derive one new column from existing column
df=pd.DataFrame(technologies)
df2 = df.assign(Discount_Percent=lambda x:x.fees*x.discount / 100)
print(df2)
#
df=pd.DataFrame(technologies)
df["MNCCompanies"]=MNCCompanies
print(df)
# Add ne column at specific position
df=pd.DataFrame(technologies)
df.insert(0,'Tutoers',tutors)
print(df)
# rename column by pandas
import pandas as pd
technologies=({'cource':["php","py","c++","c","datascience"],
               'fees':['20000','30000','40000','1200','23000'],
              'duration':['20days','30days','40days','56days','34days'],
              'discount':[123.2,234.2,456.7,643.5,453.6]
        })
df = pd.DataFrame(technologies)
df.columns
print(df.columns)
print(df.dtypes)
#rename the columns
df2=df.rename(columns = {'cource':'cours_list','fees':'fee_in_RS','discount':'schema','duration':'kalavadhi'})
print(df2.columns)
#
df2=df.rename(columns = {'cource':'cours_list'}, axis=1)
df2=df.rename(columns = {'cource':'cours_list'}, axis='columns')
#
df.rename(columns = {'cource':'cours_list'},axis=1,inplace=True)
#
rows_count=len(df.index)
rows_count
#
rows_count=len(df.axes[0])
rows_count
#
rows_count=df.shape[0]
rows_count=df.shape[1]
print(rows_count)
#
import pandas as pd
import numpy as np
data={"A":[1,2,3],
      "B":[4,5,6],
      "C":[7,8,9]
      }
df=pd.DataFrame(data)
print(df)
# addition program
def add_3(x):
  return 3+x
df2=df.apply(add_3)
df2 
#apply function on single column
def add_4(x):
  return 4+x
df["B"]=df["B"].apply(add_4)
df["B"] 
#for double column we have to use two square bracket
df[['A','B']]=df[['A','B']].apply(add_4)
df 
#
df2=df.apply(lambda x:x+10)
df2
# lambada for single column
df["A"]=df["A"].apply(lambda x:x+10)
df
# dataframe by transform
def add_3(x):
  return 3+x
df2=df.transform(add_3)
df2 
# dataframe by map
df["B"]=df["B"].map(lambda x:x+10)
print(df)
# dataframe by np.square
df["B"]=df["B"].apply(np.square)
print(df)
#direct use of numpy
df['B']=np.square(df['B'])
print(df)
#
import pandas as pd
technologies=({'cource':["php","py","xyz","c","c++","xyz","datascience","abc","abc"],
               'fees':['20000','30000','40000','1200','111','222','23000','333','444',],
              'duration':['20days','30days','40days','56days','4days','5days','34days','5days','3days',],
              'discount':[123.2,234.2,643.5,453.6,456.7,12.3,23.4,34.5,45.6]
        })
df = pd.DataFrame(technologies)
print(df)
# it is for puting original data by removing duplicates 
#df.groupby([]).sum() 
df2=df.groupby(["cource"]).sum()
print(df2)
#for two column
df2=df.groupby(["cource","fees"]).sum()
print(df2)
#by using reset
df2=df.groupby(["cource","fees"]).sum().reset_index()
print(df2)
#
import pandas as pd
technologies=({'cource':["php","py","c++","c","datascience"],
               'fees':['20000','30000','40000','1200','23000'],
              'duration':['20days','30days','40days','56days','34days'],
              'discount':[123.2,234.2,456.7,643.5,453.6]
        })
df = pd.DataFrame(technologies)
df.columns
print(df)
#get the list of all column names from headers
column_headers = list(df.columns.values)
print("column_headers:",column_headers)
#list (df) to get column headers as list
column_headers=list(df.columns)
column_headers
#list column by using df
column_headers=list(df)
column_headers
#pandas shuffle DataFrame on rows
import pandas as pd
technologies=({'cource':["php","py","c++","c","datascience"],
               'fees':['20000','30000','40000','1200','23000'],
              'duration':['20days','30days','40days','56days','34days'],
              'discount':[123.2,234.2,456.7,643.5,453.6]
        })
df = pd.DataFrame(technologies)
print(df)
# for suffle
df1=df.sample(frac=1)
print(df1)
#for reseting shuffle from 0th row
df1=df.sample(frac=1).reset_index()
print(df1)
# showing only original sequence from 0
df1=df.sample(frac=1).reset_index(drop=True)
print(df1)
#######################################################################
######################ASSIGNMENT#####################################
#####################################################################
import pandas as pd
path=r"C:\Users\santo\Downloads\Telegram Desktop\Company_Data.csv"
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


######os ##########################################################
import os 
with open("buzzers.csv") as raw_data:
    print(raw_data.read())
#
import csv 
with open("buzzers.csv") as raw_data:
    for line in csv.reader(raw_data):
        print(line)
#
import csv 
with open("buzzers.csv") as raw_data:
    for line in csv.DictReader(raw_data):
        print(line)
#
with open("buzzers.csv") as data:
    flights={}
    for line in data:
            k,v=line.split(',')
            flights[k]=v
flights
#
with open("buzzers.csv") as data:
    flights={}
    for line in data:
            k,v=line.strip().split(',')
            flights[k]=v
flights


#apply function on single column#####################################
def add_4(x):
  return 4+x
df["Age"]=df["Age"].apply(add_4)
df["Age"]

#apply function on multiple column
def add_4(x):
  return 4+x
df["Age","Education"]=df["Age","Education"].apply(add_4)
df["Age"]
 
#for double column we have to use two square bracket#################
df[['Age','Education']]=df[['Age','Education']].apply(add_4)
df
 
# lambada for single column##########################################
df["Age"]=df["Age"].apply(lambda x:x+10)
df

# dataframe by map###################################################
df["Age"]=df["Age"].map(lambda x:x+10)
print(df)

# dataframe by np.square#############################################
df["Education"]=df["Education"].apply(np.square)
print(df)

#direct use of numpy#################################################
df['Education']=np.square(df['Education'])
print(df)

#direct use of numpy ###############################################
df['Age']=np.square(df['Age'])
print(df)

# it is for puting original data by removing duplicates #############
#df.groupby([]).sum() ###############################################
df2=df.groupby(["Education"]).sum()
print(df2)

#for two column #####################################################
df2=df.groupby(["Education","Age"]).sum()
print(df2)

#by using reset #####################################################
df2=df.groupby(["Education","Age"]).sum().reset_index()
print(df2)

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
import numpy as np
newdf = pd.DataFrame(np.random.rand(334,5),index = np.arange(334,5))
 
print(newdf)

newdf.describe()

arr = np.array([[1,2,3],[4,5,6]])
print(arr)




































