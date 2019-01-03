from common_functions import get_bs_data, export_data_to_csv


page_max = 1 + 267
film_data = []

url = "https://www.ervaringen.nl/bol.com"

soup = get_bs_data(url)

reviews = soup.find_all("div", {"class": "reviews"})[0]

for review in reviews:
    review_text = review.find_all('p', {"itemprop": "reviewBody"})
    print(review_text)
    print(review)
