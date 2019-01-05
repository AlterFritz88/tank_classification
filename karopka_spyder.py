import requests
from lxml.html import fromstring
from bs4 import *
from lxml import html
import re
import random as rd
from transliterate import translit
from names_models import *

model_WWII, vectorizer_wwii = fitter_WWII()
model_modern, vectorizer_modern = fitter_modern()
model_age, vectorizer_age = fitter_age()


labels_wwII = []
with open('spisok', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels_wwII.append(line_no_spaces[:-1])
            continue

labels_modern = []
with open('modern_tech', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels_modern.append(line_no_spaces[:-1])
            continue

url = "https://karopka.ru/community/user/16588/?MODEL=468486"
def get_title(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    title = soup.find_all('title')
    title = translit(str(title[0])[7:], "ru", reversed=True)
    a = re.search(r'\b(Karopka.ru)\b', title)
    end_point = a.start() - 3
    return title[:end_point].replace(r'/', ' ')

def get_photo(url):
    photo_list = []
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    a = soup.find_all('img')
    for st in a:
        if st['src'][:14] != '/upload/resize':
            continue
        photo_list.append('https://karopka.ru' + st['src'])
    return photo_list




for i in range(286, 460):
    print(i)
    url = "https://karopka.ru/catalog/tank/?f=-1&p={0}".format(i)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')  # parse content/ page source

    for a in soup.find_all('a', {'class': 'link'}, href=True):
        url_model = 'https://karopka.ru' + a['href']
        title_model = get_title(url_model)
        print(title_model)

        what_age = model_age.predict(vectorizer_age.transform([title_model])).argmax(axis=1)[0]
        if what_age == 0:
            age = 'WWII'
            nation = labels_wwII[model_WWII.predict(vectorizer_wwii.transform([title_model])).argmax(axis=1)[0] - 1]
        else:
            age = 'Modern'
            nation = labels_modern[model_modern.predict(vectorizer_modern.transform([title_model])).argmax(axis=1)[0] - 1]
        print(age, nation)
        print()

        dirName = 'truck-link/{0}/{1}/{2}'.format(age, nation, title_model)

        if not os.path.exists('truck-link/{0}/{1}'.format(age, nation)):
            os.mkdir('truck-link/{0}/{1}'.format(age, nation))

        if not os.path.exists(dirName):
            os.mkdir(dirName)


        try:
            photo_list = get_photo(url_model)

            for photo in photo_list:
                r = requests.get(photo)
                filename = 'truck-link/{0}/{1}/{2}/{3}-{4}.jpeg'.format(age, nation, title_model,'karopka', i + rd.randint(10, 10000000))
                with open(filename, 'wb') as f:
                    f.write(r.content)
        except:
            continue


