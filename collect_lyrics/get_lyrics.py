from bs4 import BeautifulSoup
import requests
from lxml import html
import re
import json
from time import sleep
import random

# Get all the links to the songs from any artist on azlyrics.com and save them

url = "https://www.azlyrics.com/p/postmalone.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
links = soup.find_all('div', class_='listalbum-item')


link_list = []
for link in links:
    valid_link = link.find('a')['href']
    link_list.append("https://www.azlyrics.com/" + valid_link)

with open("links.json", 'w') as file:
    json.dump(link_list, file)


def save_lyrics(list):
    with open("lyrics.json", 'w') as file:
        json.dump(list, file)


def load_lyrics(my_file):
    with open(my_file, 'r') as file:
        return json.load(file)


def get_lyrics(url):
    """Get every lyrcis from the music links and save them"""
    list = load_lyrics("lyrics.json")
    rep = requests.get(url)
    page = html.fromstring(rep.content)
    xpath_expression = '/html/body/div[2]/div[2]/div[2]/div[5]'
    lyrics = page.xpath(xpath_expression)
    lyrics = lyrics[0].text_content().strip()
    lyrics = re.sub(r'\[.*?]\s*', '', lyrics)
    list.append(lyrics)
    save_lyrics(list)


Lyrics_list = []
save_lyrics(Lyrics_list)


# link_list = load_lyrics("links.json")


for link in link_list:
    sleep(random.randint(8, 20))  # Wait few seconds to avoid being blocked
    get_lyrics(link)
    print("Saved: " + link)



