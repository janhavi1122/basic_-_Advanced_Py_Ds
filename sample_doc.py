# -*- coding: utf-8 -*-

#___________________________________________________________________________________
########################## Webscrapinng ############################################
#___________________________________________________________________________________

# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
#it is going to show all the HTML content extracted
soup=BeautifulSoup(open("C:/datasets/sample_doc.html"),'html.parser')
print(soup)
#___________________________________________________________________________________

#it will show only text
soup.text
#___________________________________________________________________________________

#it is going to show all HTML contents extracted
soup.contents
#___________________________________________________________________________________

soup.find('address')
soup.find_all('address')
soup.find_all('q')
soup.find_all('b')
table=soup.find('table')
table 
for row in table.find_all('tr'):
    columns=row.find_all('td')
    print(columns)
    
#it will show all the rows except first row 
#now we want to display M.Tech which is located  in third row and second column 
#i need to give[3][2]
table.find_all('tr')[3].find_all('td')[2]
#___________________________________________________________________________________
#___________________________________________________________________________________
##########################Online web Scrap ############################################
#___________________________________________________________________________________
from bs4 import BeautifulSoup as bs
import requests
link = " https://sanjivanicoe.org.in/index.php/contact"
page =  requests.gets(link)
page

page.content

soup = bs(page.content,'html.parser')
soup
#text will be clean but not upto the expectations
#let apply prettify mtd 
print(soup.prettify())
#now data is neet and clean 
list(soup.children)
#find all contents using tab
soup.find_all('p')
#if want to extract content from 1st row
soup.find_all('p')[1].get_text()
#if want to extract content from 2nd row
soup.find_all('p')[2].get_text()

soup.find_all('div',class_='table')































