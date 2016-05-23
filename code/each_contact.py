# Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from pylab import ginput
import colorsys
import os
import matplotlib


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
            sum += stats[i]
            particle_number += 1
        else:
            stats[i] = 0

    if particle_number != 0:
        for i in range(1, 7):
            contact_numbers[i].append(float(len((np.where(stats == i))[0])) / particle_number)
        contact_numbers[0].append(1.0 - contact_numbers[1][-1] - contact_numbers[2][-1] - contact_numbers[3][-1] - contact_numbers[4][-1] - contact_numbers[5][-1] - contact_numbers[6][-1])
    else:
        for i in range(0, 7):
            contact_numbers[i].append(0)
np.savetxt('contact_numbers.txt', contact_numbers, delimiter=' ', newline='\n')


step_arr = [i for i in range(0, 101)]

# an array of parameters, each of our curves depend on a specific
# value of parameters
parameters = np.linspace(0, 6, 7)

# norm is a class which, when called, can normalize data into the
# [0.0, 1.0] interval.
norm = matplotlib.colors.Normalize(
    vmin=np.min(parameters),
    vmax=np.max(parameters))

# choose a colormap
c_m = matplotlib.cm.rainbow

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])


for i in range(0, 7):
    plt.plot(step_arr, contact_numbers[i], marker='.', color=s_m.to_rgba(i))
plt.xlabel('step')
plt.ylabel('proportion')
plt.colorbar(s_m, ticks=parameters)
plt.plot([50, 50], [0, 1], linestyle='--', color='black')
plt.savefig('contact_number_prop.eps')
