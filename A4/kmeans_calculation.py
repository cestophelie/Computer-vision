import numpy as np
import sys
from scipy.cluster.vq import kmeans
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=1000)


def vstack(descriptor_list):  # kmeans 함수 사용하기 위해 data preprocessing
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor.astype(float)))

    return descriptors


if __name__ == '__main__':
    descriptor_list = []
    siftNum = 100000
    siftName = './sift/sift/sift' + str(siftNum)
    for i in range(0, 1000):
        des = np.fromfile(siftName, dtype='ubyte')
        des = des.reshape(-1, 128)
        descriptor_list.append(des)
        siftNum = siftNum + 1
        siftName = './sift/sift/sift' + str(siftNum)
    # print('sift Name : '+str(siftName))  # 마지막 이미지까지 잘 들어갔는지 확인 용도

    descriptors = vstack(descriptor_list)

    clusters = 150  # words (cluster) 개수 설정
    voca, _ = kmeans(descriptors, clusters, 1)  # voca 가 각 클러스터의 centroid 로 (cluster, 128) 의 shape

    with open('kmeans_2015311438.npy', 'wb') as f:  # 미리 centroid 를 구해 저장. A4_compute_descriptors 에서 import
        np.save(f, voca)
