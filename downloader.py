# coding:utf-8
import requests
import bs4
import os
import time
import random

garage_list = ['mini', 'car', 'suv', 'truck', 'mb']
Garage_list = ['mini', 'smart', 'compact', 'middle', 'large', 'luxury', 'racer', 'suv', 'mpv', 'trav', 'pcar', 'mb']
car_list = ['smart', 'compact', 'middle', 'large', 'luxury', 'racer']
suv_list = ['suv', 'mpv', 'trav']


def save(url, garage_name, name, i):
    res = requests.get(url)
    imageFile = open(os.path.join('garage1/' + garage_name, name + '(' + str(i) + ')' + '.jpg'), 'wb')
    for chunk in res.iter_content(100000):
        imageFile.write(chunk)
    imageFile.close()
    return num
    # print('save!')


# os.mkdir('garage')
try:
    os.mkdir('garage1')
except:
    pass
# garage-[mini\car(smart\compact\middle\large\luxury\racer)\suv(suv\mpv\trav)\truck(pcar)\mb(mb)\bus]
num = 0
for car_type in Garage_list:
    print(car_type)
    # if car_type == 'mini':
    #     garage_name = 'mini'
    # elif car_type in car_list:
    #     garage_name = 'car'
    # elif car_type in suv_list:
    #     garage_name = 'suv'
    # elif car_type == 'pcar':
    #     garage_name = 'truck'
    # elif car_type == 'mb':
    #     garage_name = 'mb'
    garage_name = car_type
    try:
        os.mkdir('garage/' + garage_name)
    except:
        pass
    URL = 'http://product.auto.163.com/cartype/' + car_type + '/'  # 微型X车”、“轿车”、“SUV”、“公交车”、“货车”和“面包车”
    req = requests.get(URL)
    req.encoding = 'gbk'
    bs = bs4.BeautifulSoup(req.text, "lxml")
    url = bs.select('div[class="cardhd"]')
    url1 = url[0].select('h5 > a')
    url2 = url1[0].get('href')
    for i in range(0, len(url)):
        # for i in range(0, 1):
        car = bs.select('h5 > a')[i]
        name = car.getText()
        # car_url = 'http://product.auto.163.com/series/1946.html#CX001'
        car_url = 'http://product.auto.163.com' + car.get('href')
        car_req = requests.get(car_url)
        car_req.encoding = 'gbk'
        car_bs = bs4.BeautifulSoup(car_req.text, "lxml")
        try:
            Nor_Name = car_bs.select('a[class="menu_name"]')[1]
            Nor_Name = Nor_Name.getText()
            pics = car_bs.select('#car_pic')[0]
            pic = pics.select('img')
            Nor_name = Nor_Name + '-' + name
            print(Nor_name)
            for j in range(0, len(pic)):
                pic_add = pic[j].get('data-original')
                save(pic_add, garage_name, Nor_name, j + 1)
                # print(pic_add)
        except:
            try:
                pics = car_bs.select('#gainianche')[0]
                name = pics.select('div,h1,strong')[0]
                name = name.select('strong')[0]
                name = name.getText()
                print(name)
                pics = pics.select('img')
                if len(pics) > 5:
                    length = 5
                else:
                    length = len(pics)
                for k in range(1, length):
                    pic_add = pics[k].get('src')
                    save(pic_add, garage_name, name, k)
                    # print(pic_add)
            except:
                pics = car_bs.select('#chexi-pic')[0]
                Name = car_bs.select('#header2')[0]
                Name = Name.select('h1,a')[0]
                Name = Name.select('a')[0]
                Name = Name.get('title')
                # name = name.getText()
                # name = name.replace('(停产）',' ').strip()
                # print(Name)
                name = Name + '-' + name
                print(name)
                pics = pics.select('ul,li,a')[0]
                pics = pics.select('a')
                # for l in range(0,1):
                if len(pics) > 4:
                    length = 4
                else:
                    length = len(pics)
                for l in range(0, length, 2):
                    Pic = requests.get(pics[l].get('href'))
                    Pic.bs = bs4.BeautifulSoup(Pic.text, 'lxml')
                    Pic_add = Pic.bs.select('#fullscreenBox')[0]
                    pic_add = Pic_add.select('img[alt]')[2]
                    pic_add = pic_add.get('src')
                    save(pic_add, garage_name, name, l)
                    # print(pic_add)
                    # print(n + '  complete!')
        # sleep = random.randint(0, 2)
        # time.sleep(sleep)
        # print(sleep)


