
from bs4 import BeautifulSoup
import requests
try:
    from googlesearch import search
except ImportError:
    print("No module named 'google' found")


def scrape_web(searchTerm):
    # to search
    query = searchTerm + " wiki"
    
    url = ""
    for j in search(query, tld="co.in", num=1, stop=1, pause=2):
        url = j

    print(url)

    page = requests.get(url)
    
    # scrape webpage
    soup = BeautifulSoup(page.content, 'html.parser')

    object = soup.find(id="bodyContent")

    ptags = object.find_all("p")

    print(ptags[1].get_text())

    f = open("wikitraindata.txt", "w")
    f.write(ptags[1].get_text())
    f.close()