#! /usr/bin/env python
import urllib
import urllib2
import re
from bs4 import BeautifulSoup as bs
import unicodecsv as csv
import json
import pdb
import re
import sys
import os

PAGES = 100
#genres = ['sport', 'action']

genres = ['action','adventure', 'animation', 'biography', 'comedy', 'crime' , 'documentary' , 'drama' , 'family', 'fantasy', 'history', 'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi', 'short', 'sport', 'thriller', 'war', 'western']

class DataCrawler():

    def __init__(self, genre):
        self.genre = genre
        self.names = []
        self.ratings = []
        self.desc = []
        self.directors = []
        self.genres = []
        self.images = []

    def scrape(self):

        for page in range(1, PAGES+1):
            print (page)
            url = 'http://www.imdb.com/search/title?genres=' + self.genre + '&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=2406822102&pf_rd_r=10328PMYFYE70699AW2H&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_1&page=' + str(page) + '&ref_=adv_nxt'
            html = urllib2.urlopen(url)
            soup = bs(html.read(), "html.parser")

            tags = soup.find_all('h3', class_ = "lister-item-header")
            for tag in tags:
                self.names.append(tag.a.text.strip())

            tags = soup.find_all('div', class_ = "inline-block ratings-imdb-rating")
            for tag in tags:
                self.ratings.append(tag.strong.text.strip())
            
            tags = soup.find_all('span', class_ = "genre")
            for tag in tags:
                self.genres.append(tag.text.strip())

            tags = soup.find_all('p', class_ = "text-muted")
            ctr = 0
            for tag in tags:
                if ctr % 2 :
                    self.desc.append(tag.text.strip())
                ctr += 1

            tags = soup.find_all(href = re.compile('.*adv_li_dr_0.*'))
            for tag in tags:
                self.directors.append(tag.text.strip())

            '''
            imgs = soup.find_all('img', class_="loadlate")
            for img in imgs:
                self.images.append(img['loadlate'])
            print self.images
            '''
            '''
            big_imgs = soup.find_all('div', class_ = "lister-item-content")
            for x in big_imgs:
                temp_url ="http://www.imdb.com" + re.sub(r".*href\=\"(.*)\".*", r"\1", str(x.a))
                print self.get_images(temp_url)
                break
            '''

            for x in self.names:
                try:
                    x = re.sub(r" ",r"%20", str(x))
                    print (x)
                    url2 = "http://www.omdbapi.com/?t="+""+str(x)
                    self.images.append(self.poster(url2))
                except:
                    self.images.append('')

    def get_images(self, url):
        
        html = urllib2.urlopen(url)
        soup = bs(html.read(), "html.parser")
        
        img = soup.find_all('div', class_ = "poster")
        new_url = "http://www.imdb.com"+re.sub(r".*href\=\"([^\"]*)\?ref_=tt_ov_i\".*", r"\1", str(img))
        
        html2 = urllib2.urlopen(new_url)
        soup2 = bs(html2.read(), "html.parser")

        img = soup2.find_all()
        print (img)
        return new_url

    def poster(self, url):
        try:
            url2 = url
            html2 = urllib2.urlopen(url2)
            soup2 = bs(html2.read(), "html.parser")
            data = soup2.find_all(text=True)
            res =re.sub(r".*\"Response\"\:\"([^\"]*)\".*", r"\1", str(data[0]))
            print ("response")
            print (res)
            if res == "False":
                return str("")
            temp =re.sub(r".*\"Poster\"\:\"(.*)\.jpg.*", r"\1", str(data[0]))
            
            return temp+".jpg"
        except:
            return ''




if __name__ == '__main__':
    
    l = []
    for genre in genres:
        d = DataCrawler(genre)
        d.scrape()
        l.append(d)
        f = open(d.genre+'.txt', 'wb')
        for i in range(len(d.names)):
            try:
                s = ''
                s = s+''+ d.names[i]+', ' + d.ratings[i]+', '+ d.desc[i]+', '+d.directors[i]+'\n'
                f.write(s)
            except:
                pass
        f.close()
    
    print "Downloading images"
    for y in l:
        i = 1
        os.makedirs(y.genre)
        for x in y.images:
            try:
                print x
                if x == '':
                    i = i+1
                    continue
                f = open(y.genre+'/'+str(i)+'.jpg', 'wb')
                f.write(urllib.urlopen(x).read())
                f.close()
        #    print i,"done"
                i = i+1
            except:
                i = i+1

