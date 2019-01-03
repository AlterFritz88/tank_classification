import requests
from lxml.html import fromstring
from bs4 import *
from lxml import html
import re
from transliterate import translit


url = "https://karopka.ru/community/user/16588/?MODEL=468486"
def get_title(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    title = soup.find_all('title')
    title = translit(str(title[0])[7:], "ru", reversed=True)
    a = re.search(r'\b(Karopka.ru)\b', title)
    end_point = a.start() - 3
    return title[:end_point]

page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
a = soup.find_all('img')
print(len(a))
print(a[40:])
for st in a:
    if st['src'][:14] != '/upload/resize':
        continue
    print(st['src'])



'''
for i in range(0, 460):

    url = "https://karopka.ru/catalog/tank/?f=-1&p={0}".format(i)

    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')  # parse content/ page source

    for a in soup.find_all('a', {'class': 'link'}, href=True):
        print ("Found the URL:", a['href'])
'''
