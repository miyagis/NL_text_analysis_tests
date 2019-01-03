import requests
from bs4 import BeautifulSoup
import csv


def get_bs_data(url):
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data)
    return soup


def export_data_to_csv(data, file_name):
    my_file = open(file_name, "w", encoding='utf-8')
    with my_file:
        writer = csv.writer(my_file, delimiter=";")
        writer.writerows(data)

