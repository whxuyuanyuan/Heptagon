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

axis1 = 38.60
axis2 = 38.32
stepsize = 0.04
particleArea = 618.0 * 7.0 / 4.0 * 0.6934**2 / np.tan(np.pi / 7)
phi = [particleArea / (axis1 * axis2)]

for i in range(1, 51):
    axis1 -= stepsize
    axis2 -= stepsize
    phi.append(particleArea / (axis1 * axis2))

for i in range(49, -1, -1):
    phi.append(phi[i])




results = []
contact_numbers = []
for i in range(0, 7):
    contact_numbers.append([])


for step in range(0, 101):
    fileInput = open('index_interior/step_%04d' % step, 'r')
    interior_indices = []
    for line in fileInput:
        interior_indices.append(float(line))
    fileInput.close()

    stats = np.zeros(650)
    matrix = np.zeros([650, 650])
    numbers = []
    fileInput = open('../ParticleData/contact_%04d' % step, 'r')
    for line in fileInput:
        numbers = line.split()
        first = int(float(numbers[0]))
        last = int(float(numbers[1]))
        matrix[first][last] = 1
        matrix[last][first] = 1

    for i in range(0, 650):
        stats[i] = np.sum(matrix[i])
    sum = 0.0
    particle_number = 0.0

    for i in range(0, 650):
        if interior_indices.__contains__(i):
            if stats[i] == 1:
                stats[i] = 0
            if stats[i] > 1:
                sum += stats[i]
                particle_number += 1
        else:
            stats[i] = 0
    if particle_number != 0:
        results.append(sum / particle_number)
    else:
        results.append(0)
    for i in range(1, 7):
        contact_numbers[i].append(len((np.where(stats == i))[0]))
    contact_numbers[0].append(len(interior_indices) - contact_numbers[1][-1] - contact_numbers[2][-1] - contact_numbers[3][-1] - contact_numbers[4][-1] - contact_numbers[5][-1] - contact_numbers[6][-1])

data = np.transpose(np.array(results))
np.savetxt('average_contact_number.txt', data, delimiter=' ', newline='\n')
np.savetxt('contact_numbers.txt', contact_numbers, delimiter=' ', newline='\n')
plt.plot(phi[0: 50], results[0: 50], color='b', marker='o', label='Compression')
plt.plot(phi[50: 101], results[50: 101], color='r', marker='o', label='Decompression')
plt.legend(loc='upper left')
plt.ylabel('Average contact numbers')
plt.xlabel('Packing fraction')
plt.savefig('average_contacts.eps')
plt.show()
