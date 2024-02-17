# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:32:40 2023

@author: santo
"""

from bs4 import BeautifulSoup as bs
import requests
link="https://www.flipkart.com/redmi-a2-aqua-blue-64-gb/p/itm5d49be2c0a95a?pid=MOBGR4UZWNAYKSUH&lid=LSTMOBGR4UZWNAYKSUHTCCEDX&marketplace=FLIPKART&q=redmi+a2&store=tyy%2F4io&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_9_na_na_ps&otracker1=AS_QueryStore_OrganicAutoSuggest_1_9_na_na_ps&fm=search-autosuggest&iid=cff87de4-fbcf-4ee1-b0ad-81d35c22742a.MOBGR4UZWNAYKSUH.SEARCH&ppt=sp&ppn=sp&ssid=txk01rxiv40000001699243340422&qH=6569f2c697414e74"

page=requests.get(link)
page
page.content
soup=bs(page.content,'html.parser')
print(soup.prettify())
title=soup.find_all('p',class_="_2-N8xT")
title
review_title=[]
for i in range(0,len(title)):
    review_title.append(title[i].get_text())
review_title
len(review_title)
rating=soup.find_all('div',class_='_3LWZLK _1BLPMq')
rating
rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
