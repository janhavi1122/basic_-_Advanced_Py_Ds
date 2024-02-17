# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 08:24:28 2023

@author: santo
"""

def add(a,b,c):
    sum=a+b+c
    return sum 
print(add(4,5,6))

add= lambda a,b,c:a+b+c
add(4,5,6)
#
def mul(a,b,c):
    multi=a*b*c
    return multi
print (mul(1,2,3))
#
mul=lambda a,b,c:a*b*c
mul(1,2,3)
#
val =lambda *args :sum(args)
val(1,2,3,4,5,6)
val(5,6,7,8,1,2,3,4)
#
char=lambda *args: print(*args)
char("hello","python")
#
def person (name,*data):
    print(name)
    print(data)
person('sandesh',27,'pune,09876')

#
def person (name,**data):
    print(name)
    print(data)
person(name='sandesh',age=27,city='pune',pin=9876)

#
def person (name,**data):
    print(name)
    for i,j in data .items():
        print(i,j)
person('sandesh',age=27,city='pune',pin=9876)
#
val=lambda **data:sum(data.values())
val(a=1,b=2,c=3)
#
person=lambda **data:[(i,j) for i,j in data.items()]
person(name='sandesh',age=27,city='pune',pin=9876)
#
person=lambda **data:[ j for i,j in data.items()]
person(name='sandesh',age=27,city='pune',pin=9876)
#
lst1=[2,3,4,5,6,7]
sqr=lambda lst1:[i**2 for i in lst1]
print(sqr(lst1))
#




















