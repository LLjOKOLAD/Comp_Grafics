import numpy as np
from PIL import Image, ImageOps
from math import *


def dotted_line(image, x0, y0, x1, y1, count, color):
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def dotted_line_2(image, x0, y0, x1, y1, color):
    count = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
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

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color


def brezn_line(image, x0, y0, x1, y1, color):
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


# image_matrix = np.zeros((200, 200, 3), dtype = np.uint8)

# dotted_line(image_matrix, 0, 0, 200, 200, 100,255)

# dotted_line_2(image_matrix, 0, 0, 200, 200, 255)

# x_loop_line(image_matrix, 200, 200, 0, 0, 255)

# for i in range(0,13):
# brezn_line(image_matrix, 100, 100, int(100+95*cos(2*pi*i/13)), int(100+95*sin(2*pi*i/13)),255)


name = './/models//model.obj'
textureName = './/models//model_t.bmp'
image = Image.open(textureName)
texturePic = np.array(ImageOps.flip(image))

frames = []

print(texturePic[0][0])
image_height, image_width = texturePic.shape[:2]
print(image_height, image_width)
vertices = []
wares = []
normali = []
textureCoord = []
f = open(name)
line = "aaa"
while line:
    line = f.readline()
    if (line[:2] == "v "):
        dot = line.split()
        dot = dot[1:]
        vertices.append(list(map(float, dot)))
    if (line[:3] == "vn "):
        dot = line.split()
        dot = dot[1:]
        normali.append(list(map(float, dot)))
    if (line[:2] == "f "):
        ware = line.split()
        ware = ware[1:]
        a = ware[0].split("/")
        b = ware[1].split("/")
        c = ware[2].split("/")
        wares.append(a)
        wares.append(b)
        wares.append(c)
    if (line[:3] == "vt "):
        coord = line.split()
        coord = coord[1:]
        textureCoord.append(list(map(float, coord)))

f.close()

width = 1000
height = 1000


alpha1 = 0
betta1 = 0
framesAmount = 36
for fr in range(0, framesAmount):
    print(fr + 1, " frame / ", framesAmount)

    image_matrix = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image_matrix_wareframe = np.zeros((1000, 1000, 3), dtype=np.uint8)

    x_offset = 0  # Отступы, поменять если выдаёт ошибку
    y_offset = -0.04
    z_offset = 10
    x_offset_screen = 500
    y_offset_screen = 500

    alpha = alpha1 + 3
    alpha1 = alpha
    betta = betta1 + 10
    betta1 = betta
    print(betta)
    gamma = 0
    x_scale = 4500  # Масштаб, поменять если модель не видно
    y_scale = 4500

    l = np.array([0, -1, 1])

    OldMaxX = max(map(max, vertices))
    OldMinX = min(map(min, vertices))
    OldMaxY = max(map(max, vertices))
    OldMinY = min(map(min, vertices))

    alpha = alpha * np.pi / 180
    betta = betta * np.pi / 180
    gamma = gamma * np.pi / 180


    # def mapNorm(OldValue,OldMin,OldMax,NewMin,NewMax):
    #    OldRange = (OldMax - OldMin)
    #    NewRange = (NewMax - NewMin)
    #    return (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    #
    # for i in range(0,len(wares)):
    #    x0 = int((vertices[wares[i][0]-1][0] + x_offset) * x_scale)
    #    y0 = int((vertices[wares[i][0]-1][1] + y_offset) * y_scale)
    #    x1 = int((vertices[wares[i][1]-1][0] + x_offset) * x_scale)
    #    y1 = int((vertices[wares[i][1]-1][1] + y_offset) * y_scale)
    #    x0 = int(mapNorm(x0, OldMin * x_scale, OldMax * x_scale, 0, 999))
    #    y0 = int(mapNorm(y0, OldMin * y_scale, OldMax * y_scale, 0, 999))
    #    x1 = int(mapNorm(x1, OldMin * x_scale, OldMax * x_scale, 0, 999))
    #    y1 = int(mapNorm(y1, OldMin * y_scale, OldMax * y_scale, 0, 999))
    #
    #    brezn_line(image_matrix_wareframe, x0, y0, x1, y1,255)
    #
    # img = Image.fromarray(np.flip(image_matrix_wareframe), mode = 'RGB')
    # img.save(name + 'wareframe'+'.png')


    def BarCoord(x, y, x0, y0, x1, y1, x2, y2):
        denom = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
        if denom == 0:
            return -1, -1, -1
        lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denom
        lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denom
        lambda2 = 1.0 - lambda0 - lambda1

        return lambda0, lambda1, lambda2


    z_buffer = np.zeros((1000, 1000), dtype=np.float32)
    z_buffer[0:1000, 0:1000] = np.inf

    # Матрица поворота
    m1 = np.array([[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]], dtype=np.float32)
    m2 = np.array([[cos(betta), 0, sin(betta)], [0, 1, 0], [-sin(betta), 0, cos(betta)]], dtype=np.float32)
    m3 = np.array([[cos(gamma), sin(gamma), 0], [-sin(gamma), cos(gamma), 0], [0, 0, 1]], dtype=np.float32)

    R = np.matmul(np.matmul(m1, m2), m3)

    print(OldMinX, OldMaxX, OldMinY, OldMaxY)

    for i in range(0, int(len(wares)), 3):
        progress = len(wares)
        if i % 1000 == 0:
            print(i, "/", progress)

        x0 = vertices[int(wares[i][0]) - 1][0]
        y0 = vertices[int(wares[i][0]) - 1][1]
        z0 = vertices[int(wares[i][0]) - 1][2]
        x1 = vertices[int(wares[i + 1][0]) - 1][0]
        y1 = vertices[int(wares[i + 1][0]) - 1][1]
        z1 = vertices[int(wares[i + 1][0]) - 1][2]
        x2 = vertices[int(wares[i + 2][0]) - 1][0]
        y2 = vertices[int(wares[i + 2][0]) - 1][1]
        z2 = vertices[int(wares[i + 2][0]) - 1][2]

        xyz = np.matrix(np.array([[x0, x1, x2], [y0, y1, y2], [z0, z1, z2]]), dtype=np.float32)

        #xyz = np.subtract(np.matmul(np.linalg.inv(R), xyz) , np.matmul(np.linalg.inv(R), np.array([[x_offset, x_offset, x_offset], [y_offset, y_offset, y_offset], [z_offset, z_offset, z_offset]])))

        xyz = np.matmul(R, xyz) + np.matrix(
            np.array([[x_offset, x_offset, x_offset], [y_offset, y_offset, y_offset], [z_offset, z_offset, z_offset]]),
            dtype=np.float32)

        x0 = xyz[0, 0]
        y0 = xyz[1, 0]
        z0 = xyz[2, 0]
        x1 = xyz[0, 1]
        y1 = xyz[1, 1]
        z1 = xyz[2, 1]
        x2 = xyz[0, 2]
        y2 = xyz[1, 2]
        z2 = xyz[2, 2]

        v1 = np.array([x1 - x2, y1 - y2, z1 - z2])
        v2 = np.array([x1 - x0, y1 - y0, z1 - z0])

        norm = np.cross(v1, v2)

        angle_between = ((norm[0] * l[0]) + (norm[1] * l[1]) + (norm[2] * l[2]) / (
                sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]) * sqrt(
            l[0] * l[0] + l[1] * l[1] + l[2] * l[2])))

        if angle_between > 0:
            angle_between = 0

        I0 = np.dot(np.matmul(R, np.array(normali[int(wares[i][2]) - 1])), l) / (
                np.linalg.norm(np.matmul(R, np.array(normali[int(wares[i][2]) - 1]))) * np.linalg.norm(l))
        I1 = np.dot(np.matmul(R, np.array(normali[int(wares[i + 1][2]) - 1])), l) / (
                np.linalg.norm(np.matmul(R, np.array(normali[int(wares[i + 1][2]) - 1]))) * np.linalg.norm(l))
        I2 = np.dot(np.matmul(R, np.array(normali[int(wares[i + 2][2]) - 1])), l) / (
                np.linalg.norm(np.matmul(R, np.array(normali[int(wares[i + 2][2]) - 1]))) * np.linalg.norm(l))

        x0 = x_scale * x0 / z0 + x_offset_screen
        y0 = y_scale * y0 / z0 + y_offset_screen
        x1 = x_scale * x1 / z1 + x_offset_screen
        y1 = y_scale * y1 / z1 + y_offset_screen
        x2 = x_scale * x2 / z2 + x_offset_screen
        y2 = y_scale * y2 / z2 + y_offset_screen

        xmin = floor(min(x0, x1, x2))
        if (xmin < 0): xmin = 0
        ymin = floor(min(y0, y1, y2))
        if (ymin < 0): ymin = 0
        xmax = ceil(max(x0, x1, x2))
        if (xmax > 999): xmax = 999
        ymax = ceil(max(y0, y1, y2))
        if (ymax > 999): ymax = 999

        u0 = textureCoord[int(wares[i][1]) - 1][1]
        v0 = textureCoord[int(wares[i][1]) - 1][0]
        u1 = textureCoord[int(wares[i + 1][1]) - 1][1]
        v1 = textureCoord[int(wares[i + 1][1]) - 1][0]
        u2 = textureCoord[int(wares[i + 2][1]) - 1][1]
        v2 = textureCoord[int(wares[i + 2][1]) - 1][0]

        # color = [-255 * angle_between, 0 ,0]
        for j in range(xmin, xmax):
            for k in range(ymin, ymax):
                l0, l1, l2 = BarCoord(j, k, x0, y0, x1, y1, x2, y2)
                intensity = (l0 * I0 + l1 * I1 + l2 * I2)
                if (intensity > 0):
                    intensity = 0
                # color = [-255 * intensity, 0, 0]
                if (l0 >= 0 and l1 >= 0 and l2 >= 0):
                    z_val = l0 * z0 + l1 * z1 + l2 * z2
                    if (z_val < z_buffer[k][j]):
                        color = texturePic[int(image_width * (l0 * u0 + l1 * u1 + l2 * u2))][
                            int(image_height * (l0 * v0 + l1 * v1 + l2 * v2))]
                        image_matrix[k][j] = color
                        z_buffer[k][j] = z_val

    img = Image.fromarray(np.flip(image_matrix), mode='RGB')
    frames.append(img)
    img.save(name + '.png')
    # img.show()

frames[0].save(
    'homer1.gif',
    save_all=True,
    append_images=frames[1:],  # Срез, который игнорирует первый кадр.
    optimize=True,
    duration=150,
    loop=0
)
