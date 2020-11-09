import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from CV_A1_2015311438 import A1_header as fil


def main(img, newName):
    listing = []
    sigma = 1
    position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(0, 3):
        for j in range(1, 4):
            kernel = fil.get_gaussian_filter_2d(5 + 6 * i, sigma)
            temp = fil.cross_correlation_2d(img, kernel)
            textStr = str(5 + 6 * i) + 'x' + str(5 + 6 * i) + ' ' + 's=' + str(sigma)
            cv2.putText(temp, textStr, position, font, 1, (0, 0, 0), 3)
            listing.append(temp)
            sigma = sigma + 5

        sigma = 1

    result = cv2.hconcat([listing[0], listing[1], listing[2]])
    result1 = cv2.hconcat([listing[3], listing[4], listing[5]])
    result2 = cv2.hconcat([listing[6], listing[7], listing[8]])

    plt.figure('Gaussian Filtering Result')
    plt.subplot(311), plt.imshow(result, cmap='gray', aspect="auto")  # subplot (nrows,ncols, )
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.subplot(312), plt.imshow(result1, cmap='gray', aspect="auto")  # subplot (nrows,ncols, )
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.subplot(313), plt.imshow(result2, cmap='gray', aspect="auto")  # subplot (nrows,ncols, )
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)

    final = np.vstack((result, result1, result2))
    path = './result/' + newName
    cv2.imwrite(path, final)
    difference_map(img)
    plt.show()


def difference_map(img):
    # 1d cross correlation
    programStart1 = time.time()
    kernel1d = fil.get_gaussian_filter_1d(5, 1)  # horizontal kernel
    revKernel1d = np.array([kernel1d])  # vertical kernel
    revKernel1d = revKernel1d.transpose()
    # 순차적으로 vertical, horizontal 커널 적용, 1d cross correlation 함수 두 번 호출
    filtered1d = fil.cross_correlation_1d(img, revKernel1d)
    filteredSecond = fil.cross_correlation_1d(filtered1d, kernel1d)
    programFinish1 = time.time()

    # 2d cross correlation
    programStart2 = time.time()
    kernel2d = fil.get_gaussian_filter_2d(5, 1)
    dif2 = fil.cross_correlation_2d(img, kernel2d)
    programFinish2 = time.time()

    print('1D Filtering Duration (5, 1) : ' + str(round(programFinish1 - programStart1, 2)))
    print('2D Filtering Duration (5, 1) : ' + str(round(programFinish2 - programStart2, 2)))

    difference = dif2 - filteredSecond
    print()
    print('Sum of 1d, 2d cross-correlation difference : ' + str(abs(np.sum(difference))))
    cv2.imshow('Difference Map', difference)
    cv2.waitKey(0)


print('1D Gaussian Kernel (' + str(5) + ',' + str(1) + ')')
print(fil.get_gaussian_filter_1d(5, 1))
print()
print('2D Gaussian Kernel (' + str(5) + ',' + str(1) + ')')
print(fil.get_gaussian_filter_2d(5, 1))
print()
print('-----------------Image input lenna.png starting here-----------------')
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
main(img, 'part_1_gaussian_filtered_lenna.png')

print('\n\n-----------------Image input shapes.png starting here-----------------')
img2 = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
main(img2, 'part_1_gaussian_filtered_shapes.png')
