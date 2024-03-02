from email.mime import image

import numpy as np
from PIL import Image
from math import *



def dotted_line(image, x0, y0, x1, y1, count, color):
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def dotted_line_2(image, x0, y0, x1, y1, color):
    count = sqrt((x0 - x1)**2 + (y0 - y1)**2)
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color


def brezn_line(image, x0, y0, x1, y1,color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

#dotted_line(image_matrix, 0, 0, 200, 200, 100,255)

#dotted_line_2(image_matrix, 0, 0, 200, 200, 255)

#x_loop_line(image_matrix, 200, 200, 0, 0, 255)

#for i in range(0,13):
    #brezn_line(image_matrix, 100, 100, int(100+95*cos(2*pi*i/13)), int(100+95*sin(2*pi*i/13)),255)



name = './/models//natsuki.obj' #Название модели
vertices = []
wares = []
f = open(name)
line = "aaa"
while line :
    line = f.readline()
    if (line[:2]== "v " ):
        dot = line.split()
        dot = dot[1:]
        vertices.append(list(map(float,dot)))
    if (line[:2] == "f "):
        ware = line.split()
        ware = ware[1:]
        a = int(ware[0].split("/")[0])
        b = int(ware[1].split("/")[0])
        c = int(ware[2].split("/")[0])
        wares.append([a,b])
        wares.append([b,c])
        wares.append([a,c])

image_matrix = np.zeros((1000, 1000, 3), dtype = np.uint8)

x_offset = 180 # Отступы, поменять если выдаёт ошибку
y_offset = 0
x_scale = 1000 # Масштаб, поменять если модель не видно
y_scale = 1000

def mapNorm(OldValue,OldMin,OldMax,NewMin,NewMax):
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    return int((((OldValue - OldMin) * NewRange) / OldRange) + NewMin)

OldMax = max(map(max,vertices))
OldMin = min(map(min,vertices))



for i in range(0,len(wares)):
    x0 = int(vertices[wares[i][0]-1][0]*x_scale)
    y0 = int(vertices[wares[i][0]-1][1]*y_scale)
    x1 = int(vertices[wares[i][1]-1][0]*x_scale)
    y1 = int(vertices[wares[i][1]-1][1]*y_scale)
    x0 = mapNorm(x0, OldMin * x_scale, OldMax * x_scale, 0, 999) + x_offset
    y0 = mapNorm(y0, OldMin * y_scale, OldMax * y_scale, 0, 999) + y_offset
    x1 = mapNorm(x1, OldMin * x_scale, OldMax * x_scale, 0, 999) + x_offset
    y1 = mapNorm(y1, OldMin * y_scale, OldMax * y_scale, 0, 999) + y_offset

    brezn_line(image_matrix, x0, y0, x1, y1,255)


img = Image.fromarray(np.flip(image_matrix), mode = 'RGB')
img.save(name + '.png')
#img.show()