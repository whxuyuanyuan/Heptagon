# Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from pylab import ginput
import cv2
import os
from cv2 import imshow

##To test if a point is in a polygon:
def InsidePolygon(x, y, points):
    '''Return True if a coordinate (x, y) is inside a polygon defined by a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].'''
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if (p1x == p2x) or (x <= xinters):
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

PIXEL = 3.2
START = 0
D = 80.0
r0 = np.load('../tmp/r0.npy')

origin = misc.imread('../pictures/%04d_Wh.jpg' % START)
plt.imshow(origin)
p1, p2, p3, p4 = ginput(4)
plt.close()
p1 = list(p1); p2 = list(p2); p3 = list(p3); p4 = list(p4);
print(p1, p2, p3, p4)

p1[0] += D
p1[1] += D
p2[0] += D
p2[1] -= D
p3[0] -= D
p3[1] -= D
p4[0] -= D
p4[1] += D

for step in range(0, 101):
    print step
    img = misc.imread('../pictures/%04d_Wh.jpg' % step)


    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=1)
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color='blue', linewidth=1)
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], color='blue', linewidth=1)
    plt.plot([p4[0], p1[0]], [p4[1], p1[1]], color='blue', linewidth=1)

    numbers = []
    x_inner = []
    y_inner = []
    x_edge = []
    y_edge = []
    a_inner = []
    a_edge = []
    indices = []
    index = 0

    fileInput = open('../ParticleData/data_%04d' % step, 'r')
    for line in fileInput:
        numbers = line.split()
        x = float(numbers[0])
        y = float(numbers[1])
        a = float(numbers[2])
        if InsidePolygon(y, x, [p1, p2, p3, p4]):
            x_inner.append(x)
            y_inner.append(y)
            a_inner.append(a)
            indices.append(index)
        else:
            x_edge.append(x)
            y_edge.append(y)
            a_edge.append(a)
        index += 1

    plt.imshow(img)
    for itPt in range(0,len(x_inner)):
        I1=x_inner[itPt]; I2=y_inner[itPt]; A0=a_inner[itPt];
        plt.plot([I2+r0*m.cos(A0+0*2*m.pi/7),I2+r0*m.cos(A0+1*2*m.pi/7),I2+r0*m.cos(A0+2*2*m.pi/7),I2+r0*m.cos(A0+3*2*m.pi/7),I2+r0*m.cos(A0+4*2*m.pi/7),I2+r0*m.cos(A0+5*2*m.pi/7),I2+r0*m.cos(A0+6*2*m.pi/7),I2+r0*m.cos(A0+7*2*m.pi/7)],[I1+r0*m.sin(A0+0*2*m.pi/7),I1+r0*m.sin(A0+1*2*m.pi/7),I1+r0*m.sin(A0+2*2*m.pi/7),I1+r0*m.sin(A0+3*2*m.pi/7),I1+r0*m.sin(A0+4*2*m.pi/7),I1+r0*m.sin(A0+5*2*m.pi/7),I1+r0*m.sin(A0+6*2*m.pi/7),I1+r0*m.sin(A0+7*2*m.pi/7)],linewidth=0.5,color='red')
    for itPt in range(0,len(x_edge)):
        I1=x_edge[itPt]; I2=y_edge[itPt]; A0=-a_edge[itPt]/180*m.pi;
        plt.plot([I2+r0*m.cos(A0+0*2*m.pi/7),I2+r0*m.cos(A0+1*2*m.pi/7),I2+r0*m.cos(A0+2*2*m.pi/7),I2+r0*m.cos(A0+3*2*m.pi/7),I2+r0*m.cos(A0+4*2*m.pi/7),I2+r0*m.cos(A0+5*2*m.pi/7),I2+r0*m.cos(A0+6*2*m.pi/7),I2+r0*m.cos(A0+7*2*m.pi/7)],[I1+r0*m.sin(A0+0*2*m.pi/7),I1+r0*m.sin(A0+1*2*m.pi/7),I1+r0*m.sin(A0+2*2*m.pi/7),I1+r0*m.sin(A0+3*2*m.pi/7),I1+r0*m.sin(A0+4*2*m.pi/7),I1+r0*m.sin(A0+5*2*m.pi/7),I1+r0*m.sin(A0+6*2*m.pi/7),I1+r0*m.sin(A0+7*2*m.pi/7)],linewidth=0.5,color='black')
    plt.xlim([0, img.shape[1]])
    plt.ylim([0, img.shape[0]])
    plt.axis('equal')
    plt.axis('off')
    plt.title('step '+'%04d' % step)
    plt.savefig('visualization/interior_%04d' % step + '.png', dpi=250)
    plt.close()
    data = np.transpose(np.array(indices))
    np.savetxt('index_interior/step_%04d' % step, data, delimiter=' ', newline='\n')
    if step < 50:
        p1[0] += PIXEL
        p1[1] += PIXEL
        p2[0] += PIXEL
        p4[1] += PIXEL
    else:
        p1[0] -= PIXEL
        p1[1] -= PIXEL
        p2[0] -= PIXEL
        p4[1] -= PIXEL
