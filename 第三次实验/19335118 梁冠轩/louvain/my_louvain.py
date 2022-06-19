import math
import random
import collections
import numpy as np
import networkx as nx
import community as community_louvain

#读取数据
def create_graph(path):
    G = collections.defaultdict(dict)
    with open(path) as file:
        for line in file:
            nodes = line.strip().split()
            i = int(nodes[0])
            j = int(nodes[1])
            w = 1.0         #边的权重，如果数据中有则加入
            G[i][j] = w     #使用无向图，边的两个方向权重相同
            G[j][i] = w
    return G

#创建networkx的无向图，用于community_louvain的处理
class Graph:
    def __init__(self):
        self.graph = nx.Graph()     #创建无向图

    def createGraph(self, path):
        file = open(path, 'r')
        for line in file.readlines():
            nodes = line.strip().split()
            edge = (int(nodes[0]), int(nodes[1]))   #往无向图添加边
            self.graph.add_edge(*edge)
        return self.graph

#节点类，存储节点编号和社区标号
class Vertex:
    def __init__(self, vid, cid, nodes, k_in = 0):
        self._vid = vid     #节点编号
        self._cid = cid     #社区编号
        self._nodes = nodes 
        self._kin = k_in    #结点内部的边的权重，初始为0

#Louvain类，对图进行社区发现
class Louvain:
    def __init__(self, G):
        self._G = G             #图
        self._edge_num = 0      #边数量
        self._cid_array = {}    #社区编号集合
        self._vid_array = {}    #节点集合
        for vid in self._G.keys():
            self._cid_array[vid] = {vid}        #初始化每个节点作为一个社区
            self._vid_array[vid] = Vertex(vid, vid, {vid})      #节点编号即为社区编号
            self._edge_num += sum([1 for neighbor in self._G[vid].keys() if neighbor > vid])

    #根据模块度，对所有节点进行优化
    def first_step(self):
        stop = True    #用于判断循环是否结束
        random_vid = self._G.keys()
        random.shuffle(list(random_vid))    #随机访问节点
        
        while True:
            flag = True 
            #遍历所有节点 
            for i_vid in random_vid:
                i_cid = self._vid_array[i_vid]._cid     #获取该节点的社区编号
                sum_k = sum(self._G[i_vid].values()) + self._vid_array[i_vid]._kin     #计算该节点的内外边权重之和
                cid_Q = {}      #存储所有模块度增益大于0的社区编号
                
                #遍历所有与该节点相邻的节点
                for j_vid in self._G[i_vid].keys():
                    #如果相邻节点所属的社区的模块度增益大于0，则无需处理
                    j_cid = self._vid_array[j_vid]._cid
                    if j_cid in cid_Q:
                        continue
                    else:
                        #计算该社区的模块度增益
                        tot = sum([sum(self._G[k].values()) + self._vid_array[k]._kin for k in self._cid_array[j_cid]])
                        if j_cid == i_cid:
                            tot -= sum_k
                        k_v_in = sum([v for k, v in self._G[i_vid].items() if k in self._cid_array[j_cid]])
                        Q = k_v_in - sum_k * tot / self._edge_num
                        cid_Q[j_cid] = Q

                #获取模块度增益最大的社区编号
                max_cid, max_Q = sorted(cid_Q.items(), key = lambda item: item[1], reverse = True)[0]
                #模块度增益仍大于0，更改该节点的社区编号，并且继续迭代，直到模块度不再改变
                if max_Q > 0.0 and max_cid != i_cid:                  
                    self._vid_array[i_vid]._cid = max_cid       #更改该节点的社区编号                   
                    self._cid_array[max_cid].add(i_vid)         #新社区添加该节点
                    self._cid_array[i_cid].remove(i_vid)    #旧社区去除该节点
                    flag = False
                    stop = False
            if flag:
                break

        return stop

    #每个社区合并为一个新的大节点，大节点的边权重为原始社区内所有节点的边权重之和，分配新的社区
    def second_step(self):
        new_vid_array = {}
        new_cid_array = {}
        #遍历所有社区和社区内的节点，更新每个社区内部的边的权重
        for cid, vertexs in self._cid_array.items():
            #如果该社区为空，则跳过
            if len(vertexs) == 0:
                continue
            #创建一个大节点，大节点的节点编号为该社区的社区编号
            big_vertex = Vertex(cid, cid, set())
            #将该社区内的所有节点合并为一个大节点
            for vid in vertexs:
                big_vertex._nodes.update(self._vid_array[vid]._nodes)   #将社区内所有节点添加进该大节点
                big_vertex._kin += self._vid_array[vid]._kin        #计算大节点的内部的边的权重，为社区内所有小节点的和
                #遍历vid的所有邻居，如果邻居也在该社区之内，则添加
                for k, v in self._G[vid].items():
                    if k in vertexs:
                        big_vertex._kin += v / 2.0
            new_cid_array[cid] = {cid}          #初始化新的社区编号为当前的社区编号
            new_vid_array[cid] = big_vertex     #将新的大节点添加进该社区

        #根据新的节点和社区，创建新图
        new_G = collections.defaultdict(dict)
        #遍历所有社区，计算所有社区之间的边的权重
        for cid1, vertexs1 in self._cid_array.items():
            if len(vertexs1) == 0:
                continue
            for cid2, vertexs2 in self._cid_array.items():
                #避免冗余计算
                if cid2 <= cid1 or len(vertexs2) == 0:
                    continue
                edge_weight = 0.0
                #遍历cid1社区中的节点
                for vid in vertexs1:
                    #遍历该节点在cid2社区的邻居，计算cid1和cid2两个社区的边的权重
                    for k, v in self._G[vid].items():
                        if k in vertexs2:
                            edge_weight += v
                #更新两个社区之间边的权重
                if edge_weight != 0:
                    new_G[cid1][cid2] = edge_weight
                    new_G[cid2][cid1] = edge_weight

        #更新数据
        self._cid_array = new_cid_array
        self._vid_array = new_vid_array
        self._G = new_G

    #获取完成社区发现后所有社区的节点
    def get_communities(self):
        communities = []
        for vertexs in self._cid_array.values():
            if len(vertexs) != 0:
                com = set()
                for vid in vertexs:
                    com.update(self._vid_array[vid]._nodes)
                communities.append(list(com))
        return communities

    #执行社区发现
    def execute(self):
        #不停重复两个步骤，直到模块度不再改变
        while True:
            stop = self.first_step()
            if stop:
                break
            else:
                self.second_step()
        return self.get_communities()

#计算Normalized Mutual Information
def NMI(A, B, count):
    total = count
    A_ids = set(A)
    B_ids = set(B)
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            if(idA == 0 or idB == 0):
                continue
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    Hx = 0
    for idA in A_ids:
        if(idA == 0):
            continue
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        if(idB == 0):
            continue
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat

if __name__ == '__main__':
    #读取数据，创造图，完成社区发现
    path = 'data/facebook5.txt'
    G = create_graph(path)
    algorithm = Louvain(G)
    communities = algorithm.execute()

    #获取正确的社区分类
    correct_communities = np.zeros(10000)
    count = 0       #计算原来一共有多少个社区  
    file = open("data/facebook5_circle.txt") 
    line = file.readline().replace('\n','')             
    while line: 
        node = line.split(' ')
        count += len(node) - 1
        for i in range(1, len(node)):
            correct_communities[int(node[i])] = int(node[0])
        line = file.readline().replace('\n','') 
    file.close()

    #获取完成社区发现后新的社区分类
    new_communities = np.zeros(10000)
    label = 1
    for communitie in communities:
        for i in range(len(communitie)):
            new_communities[communitie[i]] = label
        label += 1

    #计算NMI
    print("NMI1:", NMI(correct_communities, new_communities, count))

    #使用community库所提供的标准louvain社区发现函数，对数据进行处理
    G = Graph().createGraph(path)
    partition = community_louvain.best_partition(G)
    
    #获取标准社区发现的社区分类
    best_communities = np.zeros(10000)
    for node in partition:
        best_communities[node] = partition[node] + 1

    #计算NMI        
    print("NMI2:", NMI(correct_communities, best_communities, count))
    