import requests

URL = "https://ndtv.com"
page = requests.get(URL)

print(page.text)