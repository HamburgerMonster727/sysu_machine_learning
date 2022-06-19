import random
import math
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei']

#kmeans聚簇分类
class KMeans(object):
    def __init__(self, k = 2, tolerance = 0.00001):
        self.k_ = k                     #类别数量
        self.tolerance_ = tolerance     #最大容许误差

    def run(self, data):
        #初始化类中心，使用随机选取
        col = data.shape[1]
        self.center_ = {}
        for i in range(self.k_):
            index = random.randint(0, data.shape[0] - 1)
            self.center_[i] = data[index]

        #迭代计算类中心
        while 1:
            self.class_ = {}    #分类后的数据
            for i in range(self.k_):
                self.class_[i] = []
            #根据当前所获得的类中心，对数据进行分类
            for feature in data:
                distances = []
                #计算该数据对所有类中心的欧式距离
                for center in self.center_:
                    distances.append(np.linalg.norm(feature[0 : col - 1] - self.center_[center][0 : col - 1]))
                #选择距离最小的类中心作为当前数据的类
                classification = distances.index(min(distances))
                #将该数据划入类中
                self.class_[classification].append(feature)

            #记录当前的类中心点
            old_center = dict(self.center_)
            #根据当前的分类，计算新的类中心点
            for i in self.class_:
                self.center_[i] = np.average(self.class_[i], axis = 0)

            #判断当前的分类是否最优化
            optimized = True
            for center in self.center_:
                old = old_center[center][0 : col - 1]
                new = self.center_[center][0 : col - 1]
                #计算新中心点与旧中心点的误差，误差过大，则说明当前分类仍未最优化
                if np.sum((new - old) / old * 100.0) > self.tolerance_:
                    optimized = False
            #如果当前分类为最优化，则停止迭代
            if optimized:
                break

#计算Normalized Mutual Information
def NMI(A,B):
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat

if __name__ == '__main__':
    #读取数据
    data = np.loadtxt('data1.txt', dtype = float, delimiter = ",")
    rgb = ['r', 'g', 'b', 'y', 'black', 'gray', 'pink', 'gold', 'purple', 'orange']
    #显示初始数据
    X = data[..., 0]
    Y = data[..., 1]
    plt.xlabel("X轴")
    plt.ylabel("Y轴")
    plt.scatter(X, Y, marker = 'x')
    plt.show()

    #进行kmeans分类
    kmeans = KMeans(k = 3)
    kmeans.run(data)

    #统计所有数据的原本类标和kmeans分类后的新类标
    old_label = np.zeros(data.shape[0])
    new_label = np.zeros(data.shape[0])
    x = 0
    for i in kmeans.class_:
        col = data.shape[1]
        distances = []
        center = kmeans.center_[i]

        #获取在该类中，离中心点距离最近的元素
        for feature in kmeans.class_[i]:
            distances.append(np.linalg.norm(feature[0 : col - 1] - center[0 : col - 1]))
        min_index = distances.index(min(distances))

        #将该元素所属的类别，作为该类所有元素的类别
        label = int(kmeans.class_[i][min_index][col - 1])
        for feature in kmeans.class_[i]:
            old_label[x] = int(feature[col - 1])
            new_label[x] = label
            x += 1

    #计算Normalized Mutual Information
    nmi = NMI(old_label, new_label)
    print("Normalized Mutual Information: ", nmi)

    #计算目标函数，平方误差
    square_error = 0
    for i in kmeans.class_:
        avg = np.average(kmeans.class_[i], axis = 0)
        for feature in kmeans.class_[i]:
            square_error += pow(np.linalg.norm(feature[0 : col - 1] - avg[0 : col - 1]), 2)
    print("Square Error: ", square_error)

    #显示分类后的数据
    for i in kmeans.class_:
        for feature in kmeans.class_[i]:
            plt.scatter(feature[0], feature[1], marker = 'x', c = rgb[i])
    #显示每个类的类中心
    for center in kmeans.center_:
        plt.scatter(kmeans.center_[center][0], kmeans.center_[center][1], marker = '*', s = 200)
    plt.show()