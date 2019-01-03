from common_functions import get_bs_data, export_data_to_csv


page_max = 1 + 267
film_data = []

url = "https://www.filmtotaal.nl/recensie?page="

for page_loop in range(1, page_max):
    url_bis = url + str(page_loop)
    soup = get_bs_data(url_bis)

    for ul in soup.find_all('ul', {'class': 'news-headline'}):
        for li in ul.find_all('li'):
            stars = li.find_all('div', {'class': 'stars medium-small review'})
            short_text = li.find_all('p', {'class': 'bite'})[0].getText()
            film_title = li.find_all('h2')[0].getText()

            star_rating = 0.0
            for star in stars:
                # print(star)
                for s in star:
                    s = str(s)
                    # print(s[1:4])
                    if s[1:5] == "span":
                        if "whole" in s:
                            star_rating += 1
                        elif "half" in s:
                            star_rating += 0.5
            a_film = (star_rating, short_text, film_title)
            film_data.append(a_film)

export_data_to_csv(file_name="film_recencies.csv", data=film_data)

# with open('all_data.csv', 'a') as fd:
#     fd.write(film_data)
