import numpy as np
import sys
import struct
from scipy.spatial import distance
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=1000)


if __name__ == '__main__':
    descriptor_list = []
    siftNum = 100000
    picNum = 1000
    siftName = './sift/sift/sift' + str(siftNum)
    for i in range(0, picNum):
        # print('sift Name : ' + str(siftName))  # 마지막 이미지까지 잘 들어갔는지 확인 용도
        des = np.fromfile(siftName, dtype='ubyte')
        des = des.reshape(-1, 128)
        descriptor_list.append(des)
        siftNum = siftNum + 1
        siftName = './sift/sift/sift' + str(siftNum)

    with open('kmeans_2015311438.npy', 'rb') as f:  # 150 개의 centroid vector. 미리 계산한 파일 import
        voca = np.load(f)
    print('numpy loaded')

    cluster = 150

    img_descriptor = np.zeros((picNum, cluster), "float32")

    for k in range(0, picNum):  # sift (img) manage
        desLen = len(descriptor_list[k])  # 해당 이미지의 feature 개수
        for i in range(0, desLen):  # 한 이미지 내에서의 feature iteration
            for j in range(0, cluster):
                euclid = distance.euclidean(voca[j].T, descriptor_list[k][i])  # 가까운 voca (centroid)에 매칭
                manhattan = distance.cityblock(voca[j].T, descriptor_list[k][i])
                if j == 0:  # 처음 min distance 초기화
                    minEuclidean = euclid
                    minManhattan = manhattan
                    flagIdx = j
                else:
                    if manhattan < minManhattan:
                        minManhattan = manhattan
                        flagIdx = j
                    if euclid < minEuclidean:
                        minEuclidean = euclid
                        flagIdx = j

            img_descriptor[k][flagIdx] += 1  # image descriptor histogram

    sum = np.sum(img_descriptor)  # normalization
    img_descriptor_normalized = img_descriptor / sum

    # print('img_descriptor : '+str(img_descriptor[0]))
    # print('img_descriptor : '+str(img_descriptor[1]))

    N = int(img_descriptor.shape[0])  # num of pictures
    D = int(img_descriptor.shape[1])  # num of k 혹은 num of (classifiable) words

    dataFormat = 'ii'  # binary 형태로 수치 데이터를 저장하기 위해 struct 사용
    data = struct.pack(dataFormat, N, D)
    # f = open('./eval/2015311438_raw.des', 'wb')  # normalize 하지 않은 파일
    # f.write(data)
    # f.write(img_descriptor)  # numpy array 는 이미 바이너리로 메모리에 저장되어 있어 그대로 write

    f1 = open('./eval/2015311438.des', 'wb')  # 이 파일이 accuracy 가 최고였으며, 제출한 파일입니다.
    f1.write(data)
    f1.write(img_descriptor_normalized)

    # f.close()
    f1.close()
