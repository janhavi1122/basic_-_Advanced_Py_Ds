# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:41:01 2023

@author: santo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:48:34 2023


"""
#________________________________________________________________


import pandas as pd
import numpy as np
technologies=({
       'Courses':["Spark","PySpark","Hadoop","Python","abap",None,"pandas","numpy"],
     'Fee':[20000,25000,np.nan,22000,20000,20000,22000,15000],
     'Discount':[11.5,8.9,23.2,6.5,17.7,16.9,4.2,9.4]  
    })

df = pd.DataFrame(technologies)
print(df)
# Quick Examples of get the Number of Rows in DataFrame
rows_count =len(df.index)
rows_count
rows_count=len(df.axes[0])
rows_count
#________________________________________________________________

# For number of columns 
column_count=len(df.axes[1])
column_count
df.size # To get the number of cells
df.shape # To get the dimensions 
df.shape[0] # To get number of rows
df.shape[1] # To get number of columns
#________________________________________________________________














