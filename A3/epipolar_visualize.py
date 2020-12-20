import cv2
import numpy as np
import random


def divCoords(M):  # 두 이미지의 match 를 나눕니다.
    coords1 = np.array(M[:, :2])
    coords2 = np.array(M[:, 2:])
    coords1 = np.int32(coords1)
    coords2 = np.int32(coords2)

    return coords1, coords2


def selectCoords(coords1, coords2):
    # 3 개의 좌표점을 고릅니다. (epipolar line에 그리는 세 개 좌표 매칭)
    length = len(coords1) - 1
    randomNum = random.randint(0, length)
    idx = [randomNum]
    c1 = []
    c2 = []
    for i in range(0, 2):
        while randomNum in idx:
            randomNum = random.randint(0, length)
        idx.append(randomNum)

    for j in range(len(idx)):
        c1.append(coords1[idx[j]])
        c2.append(coords2[idx[j]])

    c1 = np.int32(c1)
    c2 = np.int32(c2)

    return c1, c2


def epiLines(c1, c2, raw_F, img1, img2):
    # 각 이미지에 뽑힌 좌표들로 epipolar line 위의 점들을 계산합니다.
    lines1, lines2 = line_calculation(raw_F, c1, c2)
    image1 = img1
    image2 = img2

    img4, _ = epipolar_lines(image2, image1, lines1, c2, c1)  # img2
    img3, _ = epipolar_lines(image1, image2, lines2, c1, c2)  # img1

    result = np.hstack((img3, img4))
    cv2.imshow('Epipolar lines result', result)


def line_calculation(fundamentalM, c1, c2):
    line1 = []
    line2 = []

    ones = np.ones((1, c1.shape[0]))
    c1 = np.insert(c1, 2, ones, axis=1)
    c2 = np.insert(c2, 2, ones, axis=1)

    for i in range(0, len(c2)):
        line2.append(c2[i].T @ fundamentalM)  # img1

    for j in range(0, len(c1)):
        line1.append(fundamentalM @ c1[j])

    return line2, line1


def epipolar_lines(img1, img2, lines, c2, c1):
    rows, cols, _ = img1.shape
    img3 = img1.copy()
    img4 = img2.copy()

    colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255)]  # 각각 red, green, blue

    for color, line, coord1, coord2 in zip(colors, lines, c2, c1):  # parallel iteration 병렬 iteration
        x1, y1 = map(int, [0, -line[2] / line[1]])
        x2, y2 = map(int, [cols, -(line[2] + line[0] * cols) / line[1]])
        # print('map result : '+str(x1) + ' ' + str(y1))

        img3 = cv2.circle(img3, tuple(coord1), 4, color, None)
        img4 = cv2.circle(img4, tuple(coord2), 4, color, None)
        img3 = cv2.line(img3, (x1, y1), (x2, y2), color, 1)

    return img3, img4