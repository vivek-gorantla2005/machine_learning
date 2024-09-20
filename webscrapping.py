import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.amazon.in/s?k=iphone&crid=3DXZ6XTOGPV5E&qid=1726621375&sprefix=iphon%2Caps%2C251&ref=sr_pg_1"
headers = headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

webpage = requests.get(url,headers=headers).text

soup = BeautifulSoup(webpage, 'html.parser')

containers = soup.find_all('div',class_='puisg-col-inner')
for i in range(0,len(containers)):
    productname = containers[i].find('span',class_="a-size-medium a-color-base a-text-normal")
    if productname:
        print(productname.text)
