import cv2
import numpy as np
import pdb
import glob
import os

images = glob.glob("./*.jpg")

print images

count = 0

pdb.set_trace()


for i in images:
    im = cv2.imread(i)
    if im == None:
        os.remove(i)
        print "removed", i
        count += 1

print "Total removed", count



