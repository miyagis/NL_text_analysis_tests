from common_functions import get_bs_data, export_data_to_csv
import re


page_max = 1 + 21
cd_data = []

url = "http://lflmagazine.nl/cd-recensies/page/"


def get_data():
    for page_loop in range(1, page_max):
        url_bis = url + str(page_loop)
        soup = get_bs_data(url_bis)
        ul = soup.find('ul', {'class': 'blog-items mini-items clearfix'})

        for li in ul.find_all('li', {'class': 'blog-item'}):
            p = str(li.find('p'))
            print(p)
            b = p.find('"')
            if b > 0:
                p_split = p.split('"')
            else:
                p_split = p.split("'")
            p_split[2] = re.sub("[^0-9]", "", p_split[2])
            p_split[2] = p_split[2][0] + "/" + p_split[2][1]
            my_list = p_split[1], p_split[2]
            cd_data.append(my_list)
            # print(cd_data)

    export_data_to_csv(file_name="lflmagazine_recencies.csv", data=cd_data)


if __name__ == '__main__':
    get_data()
