
from bs4 import BeautifulSoup
import requests
 
# get URL
url = "https://en.wikipedia.org/wiki/Banana"
page = requests.get(url)
 
# scrape webpage
soup = BeautifulSoup(page.content, 'html.parser')

object = soup.find(id="bodyContent")

ptags = object.find_all("p")

print(ptags[1].get_text())

f = open("wikitraindata.txt", "w")
f.write(ptags[1].get_text())
f.close()