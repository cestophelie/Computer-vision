import numpy as np
from math import sqrt
from math import *


def get_gaussian_filter_1d(size, sigma):
    # 1d kernel을 return
    r = range(-int(size / 2), int(size / 2) + 1)  # 커널 사이즈. 5인 경우 인덱스는 -2, -1, 0, 1, 2
    kernel = [(1 / (sigma * sqrt(2 * pi)) * exp(-float(value) ** 2 / (2 * sigma ** 2))) for value in r]
    kernel1d = np.array(kernel)

    # 커널 elements의 합이 1에 수렴할 수 있도록 normalize 해주었습니다.
    kernel1d = kernel1d / kernel1d.sum()

    return kernel1d


def get_gaussian_filter_2d(size, sigma):
    r = range(-int(size / 2), int(size / 2) + 1)  # 총 커널 사이즈. 5인 경우 인덱스는 -2, -1, 0, 1, 2

    # 우선 1d를 구하고 transpose 한 매트릭스와 outer 함수를 사용하여 2d kernel 을 만들었습니다.
    kernel = [1 / (sigma * sqrt(2 * pi)) * exp(-float(value) ** 2 / (2 * sigma ** 2)) for value in r]
    kernel1d = np.array(kernel)

    # 커널 elements 의 합이 1에 수렴할 수 있도록 normalize 해주었습니다.
    kernel1d = kernel1d / kernel1d.sum()
    kernel2d = np.outer(kernel1d, kernel1d.transpose())

    return kernel2d


def cross_correlation_1d(img, kernel):
    # 필터가 가로인지 세로인지 여부를 판단하여 두 가지 케이스로 cross-correlation 진행합니다.
    # 패딩 또한 커널의 차수 (가로, 세로) 에 따라 필터링 되지 않은 부분만 패딩합니다.
    rows = img.shape[0]
    cols = img.shape[1]
    backup = np.zeros((rows, cols))
    loc = kernel.shape[0] // 2
    if kernel.ndim == 1:
        for i in range(0, rows):
            for j in range(loc, cols - loc):
                sample = img[i, j - loc:j + loc + 1]
                sample = sample * kernel
                backup[i][j] = sample.sum()
        for k in range(1, loc + 1):  # 5(2), 11(5), 17(8)
            # cols padding
            backup[:, loc - k] = backup[:, loc]
            backup[:, cols - loc - 1 + k] = backup[:, cols - loc - 1]

    if kernel.ndim != 1:
        for i in range(loc, rows - loc):
            for j in range(0, cols):
                sample = img[i - loc:i + loc + 1, j]
                sample = sample.flatten() * kernel.flatten()
                backup[i][j] = sample.sum()
        for k in range(1, loc + 1):  # 5(2), 11(5), 17(8)
            # rows padding
            backup[rows - k, :] = backup[rows - loc - 1, :]
            backup[loc - k, :] = backup[loc, :]

    return backup


def cross_correlation_2d(img, kernel):
    loc = kernel.shape[0] // 2  # 3->1, 5->2, 7->3
    rows = img.shape[0]
    cols = img.shape[1]
    backup = np.zeros((rows, cols))
    stdLoc = loc
    kernelFlatten = kernel.flatten()

    for i in range(loc, rows - loc):
        for j in range(loc, cols - loc):
            sample = img[i - stdLoc:i + stdLoc + 1, j - stdLoc:j + stdLoc + 1]
            sample = sample.flatten() * kernelFlatten
            backup[i][j] = sample.sum()


    # padding
    for i in range(1, stdLoc + 1):  # 5(2), 11(5), 17(8)
        backup[rows - i, :] = backup[rows - stdLoc - 1, :]
        backup[stdLoc - i, :] = backup[stdLoc, :]
        backup[:, stdLoc - i] = backup[:, stdLoc]
        backup[:, cols - stdLoc - 1 + i] = backup[:, cols - stdLoc - 1]

    return backup
