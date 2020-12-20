import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from CV_A1_2015311438 import A1_header as fil


def compute_corner_response(img):
    programStart = time.time()
    win = 5
    global name
    rows = img.shape[0]
    cols = img.shape[1]
    loc = win // 2
    response = np.zeros((rows, cols))
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    filteredX = fil.cross_correlation_2d(img, sobelX)
    filteredY = fil.cross_correlation_2d(img, sobelY)

    # Single Convariance matrix
    filteredXX = filteredX * filteredX
    filteredYY = filteredY * filteredY
    filteredXY = filteredX * filteredY

    # Second moment matrix >> matrix decomposition 을 하지 않고 determinant와 trace로 판별식을 구했습니다.
    for i in range(loc, rows - loc):
        for j in range(loc, cols - loc):
            secondMomentXX = filteredXX[i - loc: i + loc + 1, j - loc: j + loc + 1].sum()
            secondMomentYY = filteredYY[i - loc: i + loc + 1, j - loc: j + loc + 1].sum()
            secondMomentXY = filteredXY[i - loc: i + loc + 1, j - loc: j + loc + 1].sum()

            determinant = secondMomentXX * secondMomentYY - secondMomentXY * secondMomentXY
            trace = secondMomentYY + secondMomentXX
            response[i][j] = determinant - 0.04 * trace * trace

    # response 값을 정규화하는 과정입니다.
    response[response < 0] = 0
    response = (response - np.amin(response)) / (np.amax(response) - np.amin(response))  # 0 과 1 사이로 정규화

    programFinish = time.time()
    print('Compute corner response elapsed time for ' + name + ' : ' + str(round(programFinish - programStart, 2)) + ' sec')

    return response


def non_maximum_suppression_win(R, win):
    programStart = time.time()
    global filtered  # 이미지는 blur 된 버전 사용했습니다.
    rows = filtered.shape[0]
    cols = filtered.shape[1]
    loc = win // 2
    suppressed_R = np.zeros((rows, cols))

    # window를 적용해서 local max 값을 찾았습니다.
    for i in range(loc, rows - loc):
        for j in range(loc, cols - loc):
            max = np.amax(R[i - loc: i + loc + 1, j - loc: j + loc + 1])  # 해당 윈도우 내의 최대값 찾기
            if R[i][j] >= max and R[i][j] > 0.1:
                suppressed_R[i][j] = R[i][j]
                # local maxima 의 저장결과 확인장치
                # backup1[i][j][0] = 0
                # backup1[i][j][1] = 255
                # backup1[i][j][2] = 0

    programFinish = time.time()
    print('Non-maximum-suppression elapsed time for ' + name + ' : ' + str(round(programFinish - programStart, 2)) + ' sec')

    return suppressed_R


def main(winSize, image):
    global name
    global filtered
    rows = image.shape[0]
    cols = image.shape[1]
    backup = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)  # bin 값 저장 변수
    backup1 = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)  # sup 값 저장 변수
    response = compute_corner_response(image)

    # response 를 이미지로 저장할 때 값이 작아 다시 0~255 사이의 값으로 normalize 하였습니다.
    response1 = cv2.normalize(response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    response1.astype(np.uint8)
    plt.figure('RAW')
    plt.imshow(response1, cmap='gray')
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.show()
    newName = './result/part_3_corner_raw_' + name
    cv2.imwrite(newName, response1)

    for i in range(0, rows):
        for j in range(0, cols):
            if response[i][j] > 0.1:  # threshhold 0.1을 충족시키는 response 값들을 lime colour 로 세팅했습니다.
                backup[i][j][0] = 0
                backup[i][j][1] = 255
                backup[i][j][2] = 0

    plt.figure('Result')
    plt.imshow(backup.astype('uint8'))
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.show()
    newName = './result/part_3_corner_bin_' + name
    cv2.imwrite(newName, backup)

    # suppressed 된 버전 출력
    suppressed_R = non_maximum_suppression_win(response, winSize)
    # max 값 중심으로 원 그리는 부분
    for i in range(0, rows):
        for j in range(0, cols):
            if suppressed_R[i][j] != 0:
                cv2.circle(backup1, (j, i), 5, (0, 255, 0), 2)

    plt.figure('Result suppressed')
    plt.imshow(backup1.astype('uint8'))
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.show()
    newName = './result/part_3_corner_sup_' + name
    cv2.imwrite(newName, backup1)


winSize = 11
# lenna.png
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
name = 'lenna.png'
filtered = fil.cross_correlation_2d(img, fil.get_gaussian_filter_2d(7, 1.5))
# gray 이미지를 rbg 채널 이미지로 변환하기 위해 normalize
filtered = cv2.normalize(filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
filtered.astype(np.uint8)
main(winSize, filtered)
print()

# shapes.png
img2 = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
name = 'shapes.png'
filtered = fil.cross_correlation_2d(img2, fil.get_gaussian_filter_2d(7, 1.5))
filtered = cv2.normalize(filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
filtered.astype(np.uint8)
main(winSize, filtered)
