import cv2
import numpy as np
import math


def get_transformed_image(img, M):
    rows = 801
    cols = 801
    print(M)
    flag = 0

    backup = np.full((rows, cols), 255)
    transformedImg = []
    global midCoord
    midCoord = np.array([51, 56, 1])
    midX, midY, _ = M @ midCoord
    midCoord[1] = midX + 350
    midCoord[0] = midY + 345
    # print('mid : ' + str(midCoord))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            coords = np.array([i, j, 1])

            next_i, next_j, _ = M @ coords
            if img[i][j] != 0 and flag != 1:
                flag = flag + 1
            transformedImg.append([img[i][j], [next_i + 350, next_j + 345]])  # pixel value, [transformed 좌표 x, y]
    length = len(transformedImg)

    for i in range(0, length):
        backup[int(transformedImg[i][1][0])][int(transformedImg[i][1][1])] = transformedImg[i][0]

    backup = cv2.normalize(backup, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    backup.astype(np.uint8)

    # arrow 를 그리는 부분
    endCoords = (400, 0)
    startCoords = (400, 800)
    startCoords1 = (0, 400)
    endCoords1 = (800, 400)
    thickness = 2
    color = (0, 255, 0)
    backup = cv2.arrowedLine(backup, startCoords, endCoords,
                             color, thickness, tipLength=0.08)
    backup = cv2.arrowedLine(backup, startCoords1, endCoords1,
                             color, thickness, tipLength=0.08)
    cv2.imshow('image', backup)

    return M


# Main
image = cv2.imread('smile.png', cv2.IMREAD_GRAYSCALE)
M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
M = get_transformed_image(image, M)
flipFlag = 0
xFlipFlag = 0

while True:
    img = image
    key = cv2.waitKey()
    if key == ord('Q'):  # Exit. Close window onclick
        break

    # rotation transformation
    elif key == ord('R'):  # rotating clockwise by 5 degrees
        angle = np.radians(5)
        sinVal = np.sin(angle)
        cosVal = np.cos(angle)

        M = ([[cosVal, sinVal, (1 - cosVal) * 55 - sinVal * 55],  # 회전 할 때에 그래프 상 원점 기준으로 회전합니다.
              [-sinVal, cosVal, sinVal *
               50 + (1 - cosVal) * 50],
              [0, 0, 1]]) @ M

    elif key == ord('r'):  # rotating counter-clockwise by 5 degrees
        angle = np.radians(-5)
        sinVal = np.sin(angle)
        cosVal = np.cos(angle)
        M = ([[cosVal, sinVal, (1 - cosVal) * 55 - sinVal * 55],
              [-sinVal, cosVal, sinVal *
               50 + (1 - cosVal) * 50],
              [0, 0, 1]]) @ M

    # translation (shift) transformation
    elif key == ord('a'):  # left shift by 5 pixels
        M = M + ([[0, 0, 0], [0, 0, -5], [0, 0, 0]])

    elif key == ord('d'):  # right shift by 5 pixels
        M = M + ([[0, 0, 0], [0, 0, 5], [0, 0, 0]])

    elif key == ord('w'):  # upward shift by 5 pixels
        M = M + ([[0, 0, -5], [0, 0, 0], [0, 0, 0]])

    elif key == ord('s'):  # downward shift by 5 pixels
        M = M + ([[0, 0, 5], [0, 0, 0], [0, 0, 0]])

    # scale transformation (size variation)
    elif key == ord('Y'):  # x축 기준으로 enlarge
        if midCoord[1] < 401 and xFlipFlag % 2 == 0:
            M = M + ([[0.05, 0, -5], [0, 0, 0], [0, 0, 0]])
        elif midCoord[1] < 401 and xFlipFlag % 2 == 1:
            M = M + ([[-0.05, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif midCoord[1] > 401 and xFlipFlag % 2 == 0:
            M = M + ([[0.05, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif midCoord[1] > 401 and xFlipFlag % 2 == 1:
            M = M + ([[-0.05, 0, 5], [0, 0, 0], [0, 0, 0]])
        else:
            M = M + ([[0.05, 0, -2.5], [0, 0, 0], [0, 0, 0]])

    elif key == ord('y'):  # x축 방향으로 shrink
        if midCoord[1] < 401 and xFlipFlag % 2 == 0:
            M = M + ([[-0.05, 0, 5], [0, 0, 0], [0, 0, 0]])
        elif midCoord[1] < 401 and xFlipFlag % 2 == 1:
            M = M + ([[0.05, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif midCoord[1] > 401 and xFlipFlag % 2 == 0:
            M = M + ([[-0.05, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif midCoord[1] > 401 and xFlipFlag % 2 == 1:
            M = M + ([[0.05, 0, -5], [0, 0, 0], [0, 0, 0]])
        else:
            M = M + ([[-0.05, 0, 0], [0, 0, 0], [0, 0, 0]])

    elif key == ord('X'):  # y축 방향으로 enlarge
        if midCoord[0] < 401 and flipFlag % 2 == 0:
            # print('1')
            M = M + ([[0, 0, 0], [0, 0.05, -5], [0, 0, 0]])
        elif midCoord[0] < 401 and flipFlag % 2 == 1:
            # print('2')
            M = M + ([[0, 0, 0], [0, -0.05, 0], [0, 0, 0]])
        elif midCoord[0] > 401 and flipFlag % 2 == 0:
            # print('3')
            M = M + ([[0, 0, 0], [0, 0.05, 0], [0, 0, 0]])
        elif midCoord[0] > 401 and flipFlag % 2 == 1:
            # print('4')
            M = M + ([[0, 0, 0], [0, -0.05, 5], [0, 0, 0]])
        else:
            M = M + ([[0, 0, 0], [0, 0.05, -2.5], [0, 0, 0]])

    elif key == ord('x'):  # y축 방향으로 축소
        if midCoord[0] < 401 and flipFlag % 2 == 0:
            M = M + ([[0, 0, 0], [0, -0.05, 5], [0, 0, 0]])
        elif midCoord[0] < 401 and flipFlag % 2 == 1:
            M = M + ([[0, 0, 0], [0, 0.05, 0], [0, 0, 0]])
        elif midCoord[0] > 401 and flipFlag % 2 == 0:
            M = M + ([[0, 0, 0], [0, -0.05, 0], [0, 0, 0]])
        elif midCoord[0] > 401 and flipFlag % 2 == 1:
            M = M + ([[0, 0, 0], [0, 0.05, -5], [0, 0, 0]])
        else:
            M = M + ([[0, 0, 0], [0, -0.05, 0], [0, 0, 0]])

    # flips
    elif key == ord('F'):  # X 축 기준으로 flip
        xFlipFlag += 1
        M = ([[-1, 0, 101], [0, 1, 0], [0, 0, 1]]) @ M

    elif key == ord('f'):  # Y 축 기준으로 flip
        flipFlag += 1  # 홀수이면 flip 한 번
        M = ([[1, 0, 0], [0, -1, 111], [0, 0, 1]]) @ M

    elif key == ord('H'):  # go back to the initial state - identity matrix
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        flipFlag = 0

    M = get_transformed_image(img, M)
    originalM = M
