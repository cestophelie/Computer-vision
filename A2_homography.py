import cv2
import numpy as np
import math
import time
import random


def getMatches(img1, img2):
    orb1 = cv2.ORB_create()  # 첫 번째 이미지의 features extraction
    kp1 = orb1.detect(img1, None)
    kp1, des1 = orb1.compute(img1, kp1)  # 특징점, 기술자

    orb2 = cv2.ORB_create()  # 두 번째 이미지의 features extraction
    kp2 = orb2.detect(img2, None)
    kp2, des2 = orb2.compute(img2, kp2)

    rows = des1.shape[0]  # 500
    cols = des1.shape[1]  # 32
    topFeatures = []

    # hamming distance 를 구하는 부분
    for i in range(rows):  # source
        maxHamming = 0  # 거리가 가장 먼 것 각각의 feature 마다
        for k in range(rows):  # desination 의 feature
            count = 0
            for j in range(cols):
                if des1[i][j] != des2[k][j]:
                    count = count + 1
                if k == 0:
                    maxHamming = count
                    distance = count
                    queryIdx = i
                    trainIdx = k
            if count < maxHamming:
                distance = count
                queryIdx = i
                trainIdx = k  # kp2[j]
                maxHamming = count

        a = cv2.DMatch(_imgIdx=0, _queryIdx=queryIdx, _trainIdx=trainIdx, _distance=distance)
        topFeatures.append(a)

    topFeatures.sort(key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, topFeatures[:10], None, flags=2)
    cv2.imshow('matches', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return topFeatures, kp1, kp2


def normalization(coordinates):
    length = len(coordinates)
    x_mean = 0
    y_mean = 0
    maxDistance = 0

    # 1) Mean subtraction
    for i in range(length):  # mean point 구하기.
        x_mean = x_mean + coordinates[i][0]
        y_mean = y_mean + coordinates[i][1]

    Mean = [(x_mean / length), (y_mean / length)]

    # 2) distance to root 2 normalization
    for a in range(0, length):  # max distance 값 구하기
        distance = math.sqrt((coordinates[a][0] - Mean[0]) ** 2 + (coordinates[a][1] - Mean[1]) ** 2)
        if distance >= maxDistance:
            maxDistance = distance
    scale = math.sqrt(2) / maxDistance

    # 3) transformation matrix
    transMatrix = np.array([[1, 0, -Mean[0]], [0, 1, -Mean[1]], [0, 0, 1]])
    scaleMatrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    transform = scaleMatrix @ transMatrix

    normalizedCoords = []
    for i in range(length):  # 기존 값들에 scale @ translation 곱하기
        originalX = coordinates[i][0]  # 원래 좌표 값들
        originalY = coordinates[i][1]

        coords = np.array([originalX, originalY, 1])
        nextX, nextY, _ = transform @ coords  # 새로운 좌표 값들
        normalizedCoords.append([nextX, nextY])

    normalizedCoords = np.asarray(normalizedCoords)

    return transform, normalizedCoords, transMatrix, scaleMatrix


def calculate_A_matrix(srcNormalized, desNormalized):
    A = []
    for i in range(0, 9):
        srcX = srcNormalized[i][0]
        srcY = srcNormalized[i][1]
        desX = desNormalized[i][0]
        desY = desNormalized[i][1]
        A.append([-srcX, -srcY, -1, 0, 0, 0, srcX * desX, srcY * desX, desX])
        A.append([0, 0, 0, -srcX, -srcY, -1, srcX * desY, srcY * desY, desY])

    A = np.asarray(A)

    return A


def compute_homography(srcPos, desPos):  # normalized 된 좌표들로 svd 를 활용해 호모그래피를 구합니다.
    Ts, srcNormalized, trans1, scale1 = normalization(srcPos)
    Td, desNormalized, trans2, scale2 = normalization(desPos)

    A_matrix = calculate_A_matrix(srcNormalized, desNormalized)
    u, s, vh = np.linalg.svd(A_matrix, full_matrices=True)

    homography = vh[-1].reshape(3, 3)
    finalHomography = np.linalg.inv(Td) @ homography @ Ts

    return finalHomography


def compute_homography_ransac(srcP, desP, th):  # src, des 각각 100 개씩 들어옴
    length = len(srcP) - 1
    randomNum = random.randint(0, length)

    global maxInliers
    maxInliers = 0
    global numInliers
    numInliers = 0
    global finalInliers
    finalInliers = []
    global count
    count = 1
    startTime = time.time()

    while True:
        # print('Count '+str(count))  # iteration 횟수
        count += 1
        randomSrc = []
        randomDes = []
        idx = []
        if time.time() - startTime < 3:  # 3초 동안만 돌아가도록 다시 구현
            for i in range(0, 9):  # 랜덤한 인덱스 아홉 개를 뽑았습니다.
                while randomNum in idx:
                    randomNum = random.randint(0, length)
                idx.append(randomNum)
            idx.sort()
            # print('idx : ' + str(idx))  # random 인덱스 추출

            for i in range(0, 9):
                selectedIdx = idx[i]
                randomSrc.append([srcP[selectedIdx][0], srcP[selectedIdx][1]])  # 뽑힌 인덱스로 match pair 리스트 재구성
                randomDes.append([desP[selectedIdx][0], desP[selectedIdx][1]])

            randomSrc = np.asarray(randomSrc)
            randomDes = np.asarray(randomDes)

            # random 하게 뽑은 match 들로 호모그래피를 구하였습니다.
            ransacHomography = compute_homography(randomSrc, randomDes)

            # 여기부터는 3 초 동안 가장 많은 maxInlier 를 가지는 호모그래피를 구합니다.
            for i in range(0, length):
                # global inliersDest
                # global inliersList
                inliersList = []
                inliersDest = []
                originalCoords = np.array([srcP[i][0], srcP[i][1], 1])
                nextX, nextY, _ = ransacHomography @ originalCoords
                desX = desP[i][0]
                desY = desP[i][1]

                distance = math.sqrt((desX - nextX) ** 2 + (desY - nextY) ** 2)
                if distance < th:
                    numInliers += 1
                    inliersList.append([nextX, nextY])
                    inliersDest.append([desX, desY])

                # if the distance is less than 10 for example, add 'NUM OF INLIERS'

            if numInliers > maxInliers:  # 기존의 inlier 개수보다 많은 inliers 를 가진 호모그래피의 경우 업데이트 합니다.
                maxInliers = numInliers
        else:  # 시간이 3초 넘어가는 순간 break 합니다.
            finalInliers = inliersList
            break

        # 3초 안에 optimal 호모그래피를 찾고, inlier들로 호모그래피를 다시 구했습니다.
    finalInliers = np.asarray(finalInliers)
    inliersDest = np.asarray(inliersDest)

    # distance 값이 계속해서 너무 크게 나와서 나머지 부분은 구현하지 못했습니다.
    # finalHomography, _ = compute_homography(finalInliers, inliersDest)

    # return finalHomography


# ----------------------- main Normalized homography ------------------------
source = cv2.imread('cv_desk.png', cv2.IMREAD_GRAYSCALE)
destination = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

matches, srcKey, desKey = getMatches(source, destination)

srcP = []
desP = []

for i in matches[:100]:  # srcP 와 desP 에 각각의 matching 포인트들을 append 한다.
    idx1 = i.queryIdx  # desk 사진
    idx2 = i.trainIdx  # cover 사진

    x1, y1 = srcKey[idx1].pt  # desk
    x2, y2 = desKey[idx2].pt  # cover

    desP.append([x1, y1])  # desk 가 destination
    srcP.append([x2, y2])  # cover 가 destination

srcP = np.array(srcP)
desP = np.array(desP)

finalM = compute_homography(srcP[:40], desP[:40])  # distance 가 최소인 40개의 feature 로 normalize 를 진행하였습니다.

width = source.shape[1]  # desk 모양
height = source.shape[0]

img_result = cv2.warpPerspective(destination, finalM, (width, height))  # destination 이 책 사진
overlay = cv2.add(source, img_result)  # source 가 desk 사진

cv2.imshow("normalized", img_result)
cv2.imshow("warped", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------- main 2 RANSAC ---------------
compute_homography_ransac(srcP[:100], desP[:100], th=10)  # ransac 에 사용하는 매칭 개수를 100 개로 설정
'''img_result1 = cv2.warpPerspective(source, ransac_homography, (width, height))
overlay = cv2.add(destination, img_result1)

cv2.imshow("ransac", img_result1)
cv2.waitKey(0)
cv2.destroyAllWindows()'''