import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.patches as pth
from pylab import ginput
import cv2
import os
from multiprocessing import Pool

lost = [68, 89]
count = 1
for i in range(1, 304):
    if lost.__contains__(i):
        i += 1
        continue
    if i % 3 == 0:
        os.system('mv pictures/IMG_' + '%04d' % count + '.JPG pictures/' + '%04d' % (i/3) + '_Pl.jpg')
    count += 1

# select extremum points
imgWh = misc.imread('pictures/0001_Pl.jpg');
plt.imshow(imgWh)
plt.title('select 2 horizontal and vertical extremum points, then middle click and close')
p1, p2, p3, p4 = ginput(0, 0)
plt.show()
plt.close()

# extract limit coordinates:
I1m = int(p3[1])
I1M = int(p4[1])
I2m = int(p1[0])
I2M = int(p2[0])

for i in range(1, 102):
    os.system('convert pictures/' + '%04d' % i + '_Pl.jpg -crop ' + str(I2M - I2m) + 'x' + str(I1M - I1m) + '+' + str(
        I2m) + '+' + str(I1m) + ' pictures/' + '%04d' % i + '_Pl.jpg > /dev/null')

print('Done!')
