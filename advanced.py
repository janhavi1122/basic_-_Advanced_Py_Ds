
"""
Created on Tue Aug  1 08:58:09 2023

@author: santo

"""
####################ADVANCED PYTHON ##############################
#LIST COMPARASION(imp for interview)
#it is for code optimazation
lst=[]
for num in range(0,20):
    lst.append(num)
print(lst)

############################
#we can use same method in list comprehension
lst=[num for num in range(0,20)]
print(lst)

##################################
names=['dada','mama','kaka']
lst=[name.capitalize()for name in names]
print(lst)
##################################
names=['dada','mama','kaka']
lst=[name.upper()for name in names]
print(lst)
########################################
names=['DADA', 'MAMA', 'KAKA']
lst=[name.lower()for name in names]
print(lst)
########################################
#list comprehension with if statement
def is_even(num):
    return num%2==0
lst=[num for num in range(10) if is_even(num)]
print(lst)
###############################
def is_odd(num):
    return num%2==1
lst=[num for num in range(10) if is_odd(num)]
print(lst)
##############################
lst=[f"{x}{y}"for x in range(3)for y in range(3)]
print(lst)
#################################
lst=[f"{x}{y}"
     for x in range(3)
     for y in range(2)]
print(lst)
######################################
#SET COMREHENSION
set_one={x
     for x in range(5)
     }
print(lst)
###################################
#Dictionary comheresion
dict={x:x*x for x in range(4)}
print(dict) 
#############################
#Generator
#it is another way for  creating iterator(interator means 0,1,2,3)
#in a simple way
#it used keyword"yield"
#instead of returning it is defined  function
#generator is implemented using a function
#when u are use tuple comhesion 1 object will be created
#we can acess value of that object using for loop

gen=(x
    for x in range(3)
    )
print(gen)
#when u are use tuple comhesion 1 object will be created
#we can acess value of that object using for loop
#tuple not use in comhesion entity
for num in gen:
    print(num)
    
################################
gen=(x for x in range(5))
next(gen)
next(gen)
#next method-it is going to show result step by step
########################################
#Function which return multiple value
#range function-return widely
def range_even(end):#range_even is function
    for num in range (0,end,2):
        yield num #yeild used to create multiple  value
for num in range_even(8):
    print(num)
########################################
def range_odd(end):
    for num in range (1,end,2):
        yield num
for num in range_odd(12):
    print(num)
    
#########################################
#tuple generator -create object
#function can not create object
#range function used in generator
#####################################
#now instead of using for loop we can writtern own generator
gen=range_even(8)
next(gen)
next(gen)
#############################
#chaining generator
def lengths(itr):
    for ele in itr:
        yield len(ele)
def hide(itr):
    for ele in itr:
        yield ele*'*'
passwords=["not-goods","give'm-pass","00100=100"]
for password in hide(lengths(passwords)):
    print(password)
#########################################
#Enumerate
#printing list with index
lst=["milk","egg","bread"]
for index in range(len(lst)):
    print(f'{index+1}{lst[index]}')
    
################************************##########
lst=["milk","egg","bread"]
for index,item in enumerate(lst,start=1):
    print(f"{index}{item}")
#############################
import string
#pick the adjective
adjective=['sleepy','slow','smelly','wet','fat','red','orange',
           'yellow','green','blue','purple','fluffy','white','pround','brave'
           ]
noun=['apple','dinasaur','ball','toaster','goat','dragon','hammer','duck','ppanda']
import random
adjective=random.choice(adjective)
noun=random.choice(noun)
#select a number
number=random.randrange(0,100)
special_char=random.choice(string.punctuation)
#create the new secure password
passwords=adjective + noun +str(number)+ special_char
print('your new passwordis:%s'%password)
def lengths(itr):
    for ele in itr:
        yield len(ele)
def hide(itr):
    for ele in itr:
        yield ele*'*'
psw=passwords
for password in hide(lengths(psw)):
    print(password,end='')
    
#################################################
#find all of the numbers from 1-1000 that are divisible by 7

div7=[n for n in range(1,1000)if n %7==0]
print(div7)
###########################################
#Find all of the number from 1-1000 that have 3 in them
three=[n for n in range(0,1000)if '3' in str(n)]
print(three)
###################################
#count the number of spaces in the string
some_string='The slow solid swam sumptuosly the slimy swamp'
spaces=[s for s in some_string if s==' ']
print(len(spaces))

###############################
#Create a list of all the consanant in the string
#'Yellow yaks like yelling and yawning and yesturday they yodled while eating yuky yams,
sentence='''Yellow yaks like yelling and yawning and yesturday they yodled while eating yuky yams'''
result=[letter  for letter in sentence if letter not in'a,e,i,o,u," "']
print(result)
##########################################
#find the common number in two list(without using a tuple ,set)list

list_a=[1,2,3,4]
list_b=[4,3,7,2,8]
common=[a for a in list_a if a in list_b]
print(common)
#############################################
#Get only the numbers in the sentence like'in 1984 there were 13 instances of a protest with over 1000 people attending

sentence='in the sentence like in 1984 there were 13 instances of a protest with over 1000 people attending'
words=sentence.split()
result=[number for number in words if not number.isalpha()]
print(result)
#isalpha-return true if all the character are alphabet letter(a-z)

################################################
#Given number = range(20),produce a list containing
#the word 'even'if
result=[]
for n in range(20):
    if n%2==0:
        result.append('even')
    else:
        result.append('odd')
print(result)
########list comprehension
result=['even' if n%2 ==0 else 'odd' for n in range(20)]
print(result)
##########################
#produce a list of tuple consisting of only
#the matching number in these lists list_a=[1,2,3,4,5,6,7,8,9]
#lst_b=[2,7,1,12].result would look like(4,4),(12,12)

list_a=[1,2,3,4,5,6,7,8,9]
list_b=[2,7,1,12]
result=[(a,b) for a in list_a for b in list_b if a==b]
print(result)
#ASSIGNMENT Try this for same in word in two string

################################
string_a="I like mango"
string_b="I also like a mango"
result=[(a,b) for a in string_a  for b in string_b if a==b]
print(result)
#####################################
#find all the word in a string that are the less than 4 letter

sentence='On a summer day Ramnath sonar went swimming in the sun and his red skin stung'
examine=sentence.split()
result=[word for word in examine if len(word) >=4]
print(result)
################################################
#write the python program to print a specified list
#after removing the 0th ,4th,5th element
color=['Red','Green','White','Black','Pink','Yellow']
color=[x for(i,x)in enumerate(color) if i not in(0,4,5)]
print(color)
#i=index
############################################

#write python program that create a generator function 
#that yield cube of numbers from 1 to n ,accept n from the user

def cube_generator(n):
    for i in range(1,n+1):
        yield i** 3
        #accept input from user
n=int(input("Input a number: "))
#create the generator object
cubes=cube_generator(n)
#iterate over the generator and print the cube
print("cubes of numbers from 1 to",n)
for num in cubes:
    print(num)
#############################################
#write a python program to implement a generator that generates 
#random number within a given range

import random
def random_number_generator(start,end):
    while True:
        yield random.randint(start,end)
#accept input from the user
start=int(input("input the start number:"))
end=int(input("input the end number:"))
#create the generator object
random_numbers=random_number_generator(start,end)
#Generate and  print 10 random number
print("Random number between ",start,"and",end)
for _ in range(10):
    print(next(random_numbers))
###################################

#1Dimention
#2D
#3D
#4 Tensor
#write a python program to generate a 3*4*6 3D array whose each element
array=[[['*'for col in range(6)]for col in range(4)]for row in range(3)]
print(array)
#########################################################
#Write a python program to  print the number of a
#specified list after removing even number from it
num=[7,8,120,25,44,20,27,9]
num=[x for x in num if x%2!=0]
print(num)
#####################################
str1=input("Enter the First String")
str2=input("Enter the second string")
a=str1.split()
b=str2.split()
for i in a:
    if i in b:
        print(i)
 
#1st August ######################################################
#zip
# for particular name gives hole info separatelly
names=['dada','mama','kaka']
info=[9850,6032,9785]
for nm,inf in zip(names,info):
    print(nm,info)    

#use zip function using mis match ################################
#not shows extra element
names=['dada','mama','kaka','baba']
info=[9850,6032,9785]
for nm,inf in zip(names,info):
    print(nm,info) 
    
#zip longest import from itertool#################################
from itertools import zip_longest
names=['dada','mama','kaka','baba']
info=[9850,6032,9785]
for nm,inf in zip_longest(names,info):
    print(nm,info)    

#use fill value insted null or none###############################
from itertools import zip_longest
names=['dada','mama','kaka','baba']
info=[9850,6032,9785]
for nm,inf in zip_longest(names,info,fillvalue=0):
    print(nm,inf)    

#use all(),if all the values are true then it will produce output#
lst=[2,3,6,8,9] # value must be non zero
if all (lst):
    print ('all values are true')
else:
    print('uesless')

#zero  in the list then useless will be print#####################
lst=[2,3,6,8,0] # value must be non zero
if all (lst):
    print ('all values are true')
else:
    print('uesless')
    
#use any if any one is positive ################################### 
lst=[0,0,0,8,0] # if value is here it prints it have some value
if any (lst):
    print ('it have some value')
else:
    print('uesless')
###################################################################    
lst=[0,0,0,-9,0] #for negative it prints useless
if any (lst):
    print ('it have some value')
else:
    print('uesless')    
    
#use for all zero ################################################  
lst=[0,0,0,0] # all value zero then prints usless
if any (lst):
    print ('it have some value')
else:
    print('uesless')
    
#count it is itertool count the values #############################
from itertools import count
counter=count()
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))

#count start from one #############################################
from itertools import count
counter=count(start=1)
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))

#cycle() #########################################################
#repeating tasks
import itertools
instructions=("Eat","Code","Sleep")
for instruction in itertools.cycle(instructions):
    print(instruction)
    
#repeat ##########################################################
from itertools import repeat
for msg in repeat("kept pationce",times=3):
    print(msg)


# processes of selection ########################################
#permitution=related to act of arrenging all yhe members of a set 
#into some sequence or order
#combination= way of selecting items 
#from collection order does not matter

#combination ######################################################
from itertools import combinations
players=['Jhon','Jani','Janardhan']
for i in combinations(players,2):
    print(i)

#permutation #####################################################
from itertools import permutations
players=['Jhon','Jani','Janardhan']
for seat in permutations(players,2):
    print(seat)

#product ########################################################
from itertools import product
team_a=['Rohit','Pandya','Bumrah']
team_b=['Virat','Manish','sami']
for pair in product(team_a,team_b):
    print(pair)