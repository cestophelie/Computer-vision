import cv2
import numpy as np
import random
import time
from CV_A3_2015311438 import compute_avg_reproj_error as com
from CV_A3_2015311438 import epipolar_visualize as epi

# GLOBAL VARIABLES
minIdx = []
finalF = np.array([])
F = np.array([])
finalNorm = np.array([])
finalMine = np.array([])
best_model = np.array([])

# error result global variable
raw_result = 0
norm_result = 0
my_result = 0

count = 0
num = 8
normFlag = 0


def eight_point(Matches, pointNum):  # 호출 시 전달 받은 8개의 POINT 로 fundamental matrix 를 리턴합니다.
    global count
    A = np.zeros((0, 9))
    length = len(Matches) - 1
    idx = []
    eightP = []
    randomNum = random.randint(0, length)
    idx.append(randomNum)

    for i in range(0, pointNum - 1):
        while randomNum in idx:
            randomNum = random.randint(0, length)
        idx.append(randomNum)

    for j in range(0, pointNum):
        eightP.append(Matches[idx[j]])  # normalize 된 점 중 랜덤 8개 포인트로 fundamental matrix 계산
    eightP = np.asarray(eightP)

    for i in range(0, pointNum):  # eight point로 A 매트릭스를 생성합니다.
        x1 = eightP[i][0]
        y1 = eightP[i][1]
        x2 = eightP[i][2]
        y2 = eightP[i][3]

        row = np.asarray([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
        A = np.append(A, row.reshape((1, 9)), axis=0)

    u, s, v = np.linalg.svd(A, full_matrices=False)
    eightF = v[-1].reshape(3, 3).T

    return eightF


def compute_F_raw(M):
    programStart = time.time()
    count = 0
    global finalF
    global minIdx
    global F
    global raw_result

    while True:  # reprojection error 가 낮은 fundamental matrix 로 3초 동안 업데이트 합니다.
        if time.time()-programStart < 1:
            F = eight_point(M, 8)
            calculate = com.compute_avg_reproj_error(M, F)
            # print('cal : '+str(calculate))
            if count == 0:  # 첫 번째 연산인 경우 error와 fundamental matrix 업데이트
                minVal = calculate
                count = 1
                finalF = F

            if calculate < minVal:  # error 최소값 업데이트
                minVal = calculate
                finalF = F
        else:
            break

    # Global variable 값 업데이트
    raw_result = minVal

    return finalF


def compute_F_norm(M):
    programStart = time.time()
    count = 0
    global finalNorm
    global norm_result
    global num
    global normFlag

    coords1, coords2 = epi.divCoords(M)
    ones = np.ones((1, coords1.shape[0]))
    coords1 = np.insert(coords1, 2, ones, axis=1)  # (x, y, 1) 형태로 변환
    coords2 = np.insert(coords2, 2, ones, axis=1)

    x1, y1 = coords1[:, 0], coords1[:, 1]
    x2, y2 = coords2[:, 0], coords2[:, 1]

    # mean 값 구하기
    mean_x1 = np.mean(x1)
    mean_y1 = np.mean(y1)
    mean_x2 = np.mean(x2)
    mean_y2 = np.mean(y2)

    # 원점으로 부터의 distance 구하기. center 기준 가장 멀리 있는 점을 기준으로 distance 구하기
    d1 = np.amax(np.sqrt(np.square(x1 - mean_x1) + np.square(y1 - mean_y1)))
    d2 = np.amax(np.sqrt(np.square(x2 - mean_x2) + np.square(y2 - mean_y2)))

    fin_d1 = np.sqrt(2) / d1
    fin_d2 = np.sqrt(2) / d2

    # normalize transformation matrix
    transM1 = np.array([[1, 0, -mean_x1], [0, 1, -mean_y1], [0, 0, 1]])
    scaleM1 = np.array([[fin_d1, 0, 0], [0, fin_d1, 0], [0, 0, 1]])
    transform1 = scaleM1 @ transM1

    transM2 = np.array([[1, 0, -mean_x2], [0, 1, -mean_y2], [0, 0, 1]])
    scaleM2 = np.array([[fin_d2, 0, 0], [0, fin_d2, 0], [0, 0, 1]])
    transform2 = scaleM2 @ transM2

    # M 값들 normalize 합니다. 각각 이미지1, 이미지2에 해당하는 점들입니다.
    normalized1 = (transform1 @ coords1.T).T
    normalized2 = (transform2 @ coords2.T).T

    normalized1 = normalized1[:, :2]  # homogeneous 하게 바꾼 마지막 z 좌표를 normalize 한 뒤 없애줍니다.
    normalized2 = normalized2[:, :2]

    # 원래 M 형태가 이미지 1, 2 coordinates의 조합이기 때문에 두 이미지의 match 들을 다시 합쳐줍니다.
    normalized2_1 = normalized2[:, 0].reshape(1, normalized2.shape[0])
    normalized2_2 = normalized2[:, 1].reshape(1, normalized2.shape[0])
    normalized1 = np.insert(normalized1, 2, normalized2_1, axis=1)
    normalized1 = np.insert(normalized1, 3, normalized2_2, axis=1)

    while True:
        if time.time() - programStart < 1:  # 1초 동안 가장 좋은 결과값을 뽑습니다.
            normF = eight_point(normalized1, num)

            # rank 2 로 만듭니다.
            u2, s2, v2 = np.linalg.svd(normF)
            s2[2] = 0
            normF = u2 @ np.diag(s2) @ v2
            # un-normalize
            normF = transform2.T @ normF @ transform1
            # 최소의 error 를 가지는 fundamental matrix 로 업데이트합니다.
            calculate = com.compute_avg_reproj_error(M, normF)

            if count == 0:
                minVal = calculate
                count = 1
                finalNorm = normF

            if calculate < minVal:
                minVal = calculate
                # error가 최소인 최종 normalized fundamental matrix
                finalNorm = normF
        else:
            break

    # Global variable 값 업데이트
    if normFlag == 0:  # 이후 mine에서의 호출에서는 norm 결과값 업데이트 안 되도록
        norm_result = minVal

    return finalNorm


def count_inliers(F, M, thresh):
    # Euclidean 과 sampson distance 비교 결과값 sampson이 퍼포먼스가 더 좋았습니다.
    # Sampson distance 는 https://arxiv.org/pdf/1706.07886.pdf 참고
    coords1, coords2 = epi.divCoords(M)
    ones = np.ones((1, coords1.shape[0]))
    coords1 = np.insert(coords1, 2, ones, axis=1)  # (x, y, 1) 형태로 변환
    coords2 = np.insert(coords2, 2, ones, axis=1)
    reproj = []
    line1 = []
    line2 = []
    line3 = []
    line4 = []
    # distance = np.array([])

    # Euclidean distance >> sampson 보다 정확도가 높지 않았습니다.
    for i in range(len(coords1)):
        np.sqrt(np.sum(((coords2[i] - coords1[i]) ** 2))) / len(M)
    # distance = np.asarray([np.sqrt(np.sum(((coords2[i] - coords1[i]) ** 2))) / len(M) for i in np.arange(
    # coords1.shape[0])]) Sampson distance
    for i in range(len(coords1)):
        reproj.append(coords2[i].T @ F @ coords1[i])
        line1.append((F @ coords1[i])[0])
        line2.append((F @ coords1[i])[1])
        line3.append((F @ coords2[i])[0])
        line4.append((F @ coords2[i])[1])

    reproj = np.asarray(reproj)
    line1 = np.asarray(line1)
    line2 = np.asarray(line2)
    line3 = np.asarray(line3)
    line4 = np.asarray(line4)

    bottom = np.square(line1) + np.square(line2) + np.square(line3) + np.square(line4)
    distance = np.square(reproj) / bottom
    distance = distance < thresh

    return distance


def selected_group(M, inliersList):
    updated_M = []

    for i in range(len(inliersList)):
        if inliersList[i]:
            updated_M.append(M[i])
    updated_M = np.asarray(updated_M)

    return updated_M


def compute_F_mine(M):
    # normalized 에서 컨센서스가 최대인 케이스를 구하기 위해 ransac 알고리즘을 더했습니다.
    # 좌표 20개로 eight point algorithm 에 대입하여 정확도가 올라갔습니다.
    programStart = time.time()
    global num
    global my_result
    global finalMine
    global normFlag

    num = 20
    maxInliers = 0
    normFlag = 1
    thresh = 0.07
    inliersNum = 0
    mine_F = compute_F_norm(M)
    inliersList = count_inliers(mine_F, M, thresh)

    for i in range(len(inliersList)):
        if inliersList[i]:
            inliersNum = inliersNum + 1

    while True:
        if time.time() - programStart < 3:
            if inliersNum > maxInliers:
                maxInliers = inliersNum
                selected = selected_group(M, inliersList)
                finalMine = compute_F_norm(selected)
        else:
            break
    calculate = com.compute_avg_reproj_error(M, finalMine)
    my_result = calculate
    num = 8
    normFlag = 0
    # print('fin')
    return mine_F


def printResult(name1, name2):
    global raw_result
    global norm_result
    global my_result
    print('Average reprojection errors (' + name1 + ' and ' + name2 + ')')
    print('    Raw = '+str(raw_result))
    print('    Norm = ' + str(norm_result))
    print('    Mine = ' + str(my_result))


def visualize(raw_F, norm_F, mine_F, name1, name2, img1, img2):
    printResult(name1, name2)  # 콘솔에 raw, norm, mine 결과값 프린트

    # --------------Mine visual result-----------------
    # feature 좌표를 이미지 1과 이미지 2 에 해당하는 좌표들 두 array 로 나눕니다.
    coords1, coords2 = epi.divCoords(M)
    # random 한 인덱스 세 개를 고릅니다.
    c1, c2 = epi.selectCoords(coords1, coords2)
    # epipolar line을 계산해서 draw lines 함수 호출
    epi.epiLines(c1, c2, mine_F, img1, img2)

    # onclick 'q' 다른 랜덤 조합을 보여줍니다.
    while True:
        key = cv2.waitKey()
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            # 위 과정 반복
            c1, c2 = epi.selectCoords(coords1, coords2)
            # epipolar line을 계산해서 draw lines 함수 호출
            epi.epiLines(c1, c2, mine_F, img1, img2)


if __name__ == '__main__':
    # ----------------Temple----------------
    temple1 = cv2.imread('temple1.png', cv2.IMREAD_COLOR)
    temple2 = cv2.imread('temple2.png', cv2.IMREAD_COLOR)
    M = np.loadtxt('temple_matches.txt')  # shape is (110, 4)
    name1 = 'temple1.png'
    name2 = 'temple2.png'
    raw_F = compute_F_raw(M)
    norm_F = compute_F_norm(M)
    mine_F = compute_F_mine(M)
    visualize(raw_F, norm_F, mine_F, name1, name2, temple1, temple2)
    print()

    # ----------------House----------------
    house1 = cv2.imread('house1.jpg', cv2.IMREAD_COLOR)
    house2 = cv2.imread('house2.jpg', cv2.IMREAD_COLOR)
    M = np.loadtxt('house_matches.txt')  # shape is (110, 4)
    name1 = 'house1.jpg'
    name2 = 'house2.jpg'
    raw_F = compute_F_raw(M)
    norm_F = compute_F_norm(M)
    mine_F = compute_F_mine(M)
    visualize(raw_F, norm_F, mine_F, name1, name2, house1, house2)
    print()

    # ----------------Library----------------
    library1 = cv2.imread('library1.jpg', cv2.IMREAD_COLOR)
    library2 = cv2.imread('library2.jpg', cv2.IMREAD_COLOR)
    M = np.loadtxt('library_matches.txt')  # shape is (110, 4)
    name1 = 'library1.jpg'
    name2 = 'library2.jpg'
    raw_F = compute_F_raw(M)
    norm_F = compute_F_norm(M)
    mine_F = compute_F_mine(M)
    visualize(raw_F, norm_F, mine_F, name1, name2, library1, library2)