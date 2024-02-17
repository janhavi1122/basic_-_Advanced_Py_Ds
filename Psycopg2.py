# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 08:36:43 2023

@author: santo
"""

import psycopg2 as pg2
#___________________________________________________________________________________

#create connection with prostgresql
#'password' is whatever password you set 
conn=pg2.connect(database='dvdrental',user='postgres',password='student')
#___________________________________________________________________________________

#establist connection and start cursor to be ready to query
cur = conn.cursor()
#___________________________________________________________________________________
#pass prosgresql query
cur.execute("select * from payment")
#___________________________________________________________________________________
#return a tuple of the first row
cur.fetchone()
#___________________________________________________________________________________
#return n no of row
cur.fetchmany(10)
#___________________________________________________________________________________
#return all rows at onece
cur.fetchall()
#___________________________________________________________________________________
#to save and index result, assign it to a variable
data=cur.fetchmany(10)
#___________________________________________________________________________________
conn.close()
for row in rows:
    print(row)
#___________________________________________________________________________________



























