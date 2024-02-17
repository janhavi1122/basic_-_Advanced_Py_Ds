# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:59:34 2023

@author: santo
"""

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
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:56:44 2023

@author: santo
"""
#PRACTICE
#a into int
a=9
print(type(a))

#b into float
b=3.0 
print(type(b))

#c into float addition
c=a+b
print(c)
print(type(c))

#c into float multiplication
c=a*b
print(c)
print(type(c))

#c into float multiplication
c=a/b
print(c)
print(type(c))



Tuple1 = tuple('Geeks')
print(Tuple1)

L1=list((1,2,3,4,5))
print(L1)



#________________________________________________________________________________

lst1 =[2,34,56,3,7]
lst2 =[34,8,9,0,6]
def function(lst1,lst2):
    if set(lst1)|set(lst2):
        return True
function(lst1,lst2)
#________________________________________________________________________________

lst3= [i+6 for i in  lst1]
lst3
#________________________________________________________________________________

str = 'JANUHERE'
str[::-1]

#________________________________________________________________________________

dict ={
       'Stud':'manoj',
       'class': 'TY',
       'Division':'A'}
df=dict.items()
print(df)
#________________________________________________________________________________

dict1={
       'a':9867,
       'b':765,
       'c':987,
       'd':7800
       }
dict2 = {x:val for x,val in dict1.items() if val > 2000}
dict2
#________________________________________________________________________________


fname ='data.txt'
with open(fname, mode ='rb') as f:
    contents = f.read()
    print(type(contents))
    print(contents)
    print(contents.decode('utf8'))
    
#________________________________________________________________________________
    
import itertools as it
start=10
step=10
counter=it.count(start,step)
for i in counter:
    print(i)



#________________________________________________________________________________

even = lambda x: x%2==0
odd = lambda x: x%2 !=0
lst1=[1,2,3,4,5,6,7,8,9,10]

list(filter(even,lst1))
list(filter(odd,lst1))
#________________________________________________________________________________

import pandas as pd
dict1={
       'name':['pallavi','janhavi','vaishnavi'],
       'score':[89,99,87],
       'attempt':[1,2,3],
       'qualify':['Y','Y','Y']
       }
row_lebals=['A','B','C']
df=pd.DataFrame(dict1,index=row_lebals)
print(df)
#________________________________________________________________________________

import matplotlib.pyplot as plt
import numpy as np
plt.plot([0,1,2,3,4])
#________________________________________________________________________________

#1.Write a Python program to sum all the items in a list.
lst1=[9,8,7,6]
sum(lst1)

#2.Create an identical list from the first list using list comprehension.
lst1=[1,2,3,4]
lst2=[i for i in lst1]
print(lst2)


#3.Write a Python function to multiply all the numbers in a list.
def mul(lst):
    prod=1
    for i in lst1:
        prod*=i
    return prod
lst1=[1,2,3,4]
mul(lst1)

#4.Write a Python script to sort (ascending and descending) a dictionary by value.


#5.First, create a range from 100 to 160 with steps of 10 using dict comprehension
rng=range(100,160,10)
print(list(rng))

#6.Write a Python program to read an entire text file.
txt = open('C:/Users/santo/.spyder-py3/autosave/pid9520.txt')
txt.read()

#Create a program that asks to user to enter their name and age print out a msg addressed  to them that tells them year that they will turn 100 year old
import datetime
name = input("Enter your name=")
age = int(input("Enter your age="))
currentyear=datetime.datetime.now().year
print(currentyear)
dob=currentyear - age
print(dob + 100)

#even and odd 
n=int(input("Enter a number="))
if n%2==0:
    print(n,"Is Even")
else:
    print(n,"Is odd")

#fibonacci series
n=int(input("Enter a number="))
first=0
second=1
for i in range(n):
    print(first)
    temp=first
    first=second
    second=second+temp

#Reverse number
n=int(input("Enter a number="))
reverse=0
while(n>0):
    reminder=n%10
    reverse=(reverse*10)+reminder
    n=n//10
print("Reverse no=",reverse)

#Armstrong and Palindrome
n=int(input("Enter a number="))
trmp=n
result=0
while(temp>0):
    reminder=temp%10
    result=reminder**3+result
    temp=temp//10
if (result==n):
    print(result,"Armstrong No")
else:
    print(result,"Not an Armstrong No")


#palindrome
n=int(input("Enter a number="))
temp=n
reverse=0
while(temp>0):
    reminder=temp%10
    result=reverse*10+reminder
    temp=temp//10
if (temp==reverse):
    print(temp,"palindrome No")
else:
    print(temp,"Not an palindrome  No")















