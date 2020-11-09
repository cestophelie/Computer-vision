import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from CV_A1_2015311438 import A1_header as fil


# Notice
# 1. Sobel filter 는 cross correlation 필터를 사용하였습니다.


def compute_image_gradient(image):
    # cross correlation 함수를 사용하기 때문에 Sobel filter 도 cross correlation 필터를 사용하였습니다.
    programStart = time.time()
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # header 파일의 함수들을 이용하여 cross correlation 을 했습니다.
    filtered1 = fil.cross_correlation_2d(image, sobelX)
    filtered2 = fil.cross_correlation_2d(image, sobelY)

    # magnitude 를 구현하고 0~255 사이의 값으로 정규화 해주었습니다.
    mag = np.hypot(filtered1, filtered2)
    mag = mag / mag.max() * 255

    # direction 구현
    direction = np.arctan2(filtered2, filtered1)

    programFinish = time.time()
    print('Compute_image_gradient elapsed time for ' + name + ' : ' + str(
        round(programFinish - programStart, 2)) + ' sec')

    return mag, direction


def non_maximum_suppression_dir(mag, direction1):
    programStart = time.time()
    rows, cols = mag.shape
    backup = np.zeros((rows, cols), dtype=np.int32)

    # 기존의 direction은 -3.14 ~ 3.14 까지의 radius로 구현되어 있어서 각도로 바꾸었습니다.
    direction = np.rad2deg(direction1)
    # 180도 대치되는 값들은 동일한 연산을 진행하기 때문에 음수 값들에는 180도를 더해주었습니다.
    direction[direction < 0] += 180
    degree = 360
    for i in range(rows):
        for j in range(cols):
            # 사진 범위가 안 넘으면
            if ((j + 1) < cols) and ((j - 1) >= 0) and ((i + 1) < rows) and ((i - 1) >= 0):
                # 0 degrees >> 세로 edge 처리
                if (direction[i][j] >= 15 * degree / 16 or direction[i][j] < 1 * degree / 16) or (
                        7 * degree / 16 <= direction[i][j] < 9 * degree / 16):
                    if mag[i][j] >= mag[i][j + 1] and mag[i][j] >= mag[i][j - 1]:
                        backup[i][j] = mag[i][j]

                # 45 degrees 오른쪽 대각선 처리
                if (1 * degree / 16 <= direction[i][j] < 3 * degree / 16) or \
                        (9 * degree / 16 <= direction[i][j] < 11 * degree / 16):
                    if mag[i][j] >= mag[i - 1][j - 1] and mag[i][j] >= mag[i + 1][j + 1]:
                        backup[i][j] = mag[i][j]

                # 90 degrees >> 가로 edge 처리
                if (3 * degree / 16 <= direction[i][j] < 5 * degree / 16) or (
                        11 * degree / 16 <= direction[i][j] < 13 * degree / 16):
                    if mag[i][j] >= mag[i - 1][j] and mag[i][j] >= mag[i + 1][j]:
                        backup[i][j] = mag[i][j]

                # 135 degrees 왼쪽 대각선 처리
                if (5 * degree / 16 <= direction[i][j] < 7 * degree / 16) or (
                        13 * degree / 16 <= direction[i][j] < 15 * degree / 16):
                    if mag[i][j] >= mag[i + 1][j - 1] and mag[i][j] >= mag[i - 1][j + 1]:
                        backup[i][j] = mag[i][j]

    programFinish = time.time()
    print('Non_maximum_suppression_dir elapsed time for ' + name + ' : ' + str(
        round(programFinish - programStart, 2)) + ' sec')

    return backup


def main(img, name):
    filtered = fil.cross_correlation_2d(img, fil.get_gaussian_filter_2d(7, 1.5))
    mag, dir = compute_image_gradient(filtered)
    backup = non_maximum_suppression_dir(mag, dir)

    plt.figure("RAW_MAG")
    plt.imshow(mag, cmap='gray', aspect="auto")  # subplot (nrows,ncols, )
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.show()
    newName = './result/part_2_edge_raw_' + name
    cv2.imwrite(newName, mag)

    plt.figure("SUPPRESSED MAG")
    plt.imshow(backup, cmap='gray', aspect="auto")  # subplot (nrows,ncols, )
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.show()
    newName = './result/part_2_edge_sup_' + name
    cv2.imwrite(newName, backup)


# lenna.png
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
name = 'lenna.png'
main(img, name)
print()

# shapes.png
img2 = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
name = 'shapes.png'
main(img2, name)
