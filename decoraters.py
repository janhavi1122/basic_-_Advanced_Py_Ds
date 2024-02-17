# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:17:21 2023

@author: santo
"""
def plus_one(num):
    num1=num+1
    return num1
plus_one(5)

def plus_one(num):
  
    def add_one(num):
        num1=num+1
        return num1
    result =add_one(num)
    return result
plus_one(6)
#
def plus_one(num):
    result1=num+1
    return result1


def function_call(function):
    result=function(5)
    return result
function_call(plus_one)
#
def hello_function():
    def say_he():
        return"babyyyyy"
    return say_he
hello=hello_function()
hello
#
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:13:16 2023

@author: hp
"""

#function(methond)

def prime_num(num):
    for i in range(2,num):
        if (num%i==0):
           return "The number is not prime"
           break
    return "The number is prime number"

print(prime_num(12))
###################################
num=int(input("enter the number:"))

for i in range(2,num):
    if num%i==0:
        print("not prime")
        break
    else:
        print("prime")
##################function without arragument
def greet_user():
#"""Display a simple greeting."""
    print("Hello!")
greet_user()
############
#positional argument
def greet_user(username):
#"""Display a simple greeting."""
    print(f"Hello,{username}")
greet_user('Sanjivani AI')
#################################
#key words

def describe_pet(animal_type,pet_name):
    print(f"\nI have a {animal_type}.")
    print(f"My {animal_type}'s name is {pet_name}.")
describe_pet('Dog','Moti')
    
##############
#Default value Argument
#when we write the function we can define default value for each parameter
def describe_pet(pet_name,animal_type='dog'):
    print(f"\nI have a {animal_type}.")
    print(f"My {animal_type}'s name is {pet_name.title()}.")
describe_pet('Moti')

################
#Avoiding Argument Factor
def describe_pet(animal_type,pet_name):
    print(f"\nI have a {animal_type}.")
    print(f"My {animal_type}'s name is {pet_name.title()}.")
describe_pet()

#################
#Roller coaster
print("Welcome to the roller coaster")
height=int(input("Please enter your height in cm"))
if height >=120:
    print("You are eligible for roller coaster")
    age=int(input("Enter your age in year"))
    if age <18:
        print('ticket is RS. 20')
    elif age>18:
        print('ticket is RS.50')
    elif age<12:
        print('ticket is RS.10')
elif age >= 12 and age <=20 :
   print('ticket is RS.15')
else:
    print("thanks")
   ########################

def days_left(age):
    age_left=80-age
    days=365*age_left
    weeks=52*age_left
    months=12*age_left
    return f"You have {days} days, {weeks} weeks, {months} months remained"
age=int(input("Enter your age"))
print(days_left(age))

####################
print("Welcome to the roller coaster")
height=int(input("Please enter your height in cm"))
bill=0
if height>=120:
    print("you are eligible for roller coster ")
    age=int(input("please enter your age"))
    if age<12:
        print("child ticket are $5")
        bill=5
    elif age<=18:
        print("Youth ticket are$7")
        bill=7
    else:
        print("Adult ticket are $12")
        bill=12
    want_photo=input("Do you want photo Y OR N")
    if want_photo=='Y':
        bill+=3
        print(f"your total bill will be {bill}")
    else:
        print(f"you total bill {bill}")
        
#################
users=["admin","employee","manager","worker","staff"]
for user in users:
    if user=="admin":
        print("Hello admin,would you like to see a status report?")
    elif user=="employee":
        print("hello employee")
    elif user=="manager":
        print("hello manager")
    elif user=="worker":
        print("hello worker")
    else:
        print("hello")
        
##########################
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
password=adjective + noun +str(number)+ special_char
print('your new passwordis:%s'%password)

##############
#we can use while loop
print('welcome to password picker')
while True:
    adjective=random.choice(adjective)
    noun=random.choice(noun)
    number=random.randrange(0,100)
    special_char=random.choice(string.punctuation)
    password=adjective + noun + str(number) + special_char
    print('Your new password is:%s '% password)
    response=input('would you like to generate another password? Type Y or N')
    if response=='N':
        break
    
#####################
#write python code to determine whether the password are good or bad we
#we define strog password 1.it must have at least 8 character
#2.at least one upper case

def checkpassword(password):
    has_upper=False
    has_lower=False
    has_num=False
    for ch in password:
        if ch>='A' and ch<='Z':
            has_upper=True
        elif ch>='a' and ch<='z':
            has_lower=True
        elif ch>='0' and ch<='9':
            has__num=True
    if len(password)>=8 and has_upper and has_lower and has_num:
        return True
    else:
        return False
p=input("Enter a password  :")
if checkpassword(p):
    print("strong password")
else:
    print("Weak password")
#############
#pizza order
print("Welcome to pizza hut")
size_pizza=(input("Enter pizza size"))
add_pepp=(input("Do you want to add pepp"))
add_chees=(input("Do you want extra chees"))
bill=0
if(size_pizza=="small"):
    bill = 15
elif(size_pizza=="medium"):
    bill = 20
elif(size_pizza=="large"):
    bill = 25
if(add_pepp=='y' and size_pizza=="small"):
        bill+=2
if(add_pepp=='y' and (size_pizza=="medium") or (size_pizza=="large")):
        bill+=3
if(add_chees=="y"):
        bill+=1
print(f"Your bill is {bill}")

####################
#return value
def get_formatted_name(first_name,last_name):
    full_name=f"{first_name} {last_name}"
    return full_name
musician=get_formatted_name('Aishwarya','Sonawane')
print(musician)

#retuning a dictionary
def build_person(first_name,last_name):
    person={'first':first_name,'last':last_name}
    return person
musician=build_person('Aishwarya','Sonawane')
print(musician)

###########################################################
#passing list
#useful to pass list function
#name,number,more complex
def greet_users(names):
    for name in names:
        msg=f"Hello,{name.title()}!"
        print(msg)
usernames=['mukta','aishwarya','ankita','pallavi']
greet_users(usernames)

    
    
    
    


    
    
    
    
    


    

    
    




















