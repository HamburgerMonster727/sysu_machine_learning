import random
import operator

class UserBasedCF:
    def __init__(self, _k): 
        self.similars = {}      #用户相似度矩阵
        self.item_users = {}    #物品包含的用户
        self.user_items = {}    #用户包含的物品
        self.user_ratings = {}  #用户的评分
        self.total_items = []   #总共的物品列表
        self.k = _k         

    def load_data(self, path):
        with open(path, 'r') as f:
            for i, line in enumerate(f, 0):
                if i != 0:  
                    line = line.strip('\n')
                    user, item, rating, timestamp = line.split(',')
                    self.item_users.setdefault(item, [])
                    self.item_users[item].append(user) 
                    self.user_items.setdefault(user,[])
                    self.user_items[user].append(item)
                    self.user_ratings.setdefault(user,{})
                    self.user_ratings[user].setdefault(item, 0.)
                    self.user_ratings[user][item] = float(rating)
                    self.total_items.append(item)
        self.total_items = list(set(self.total_items))  #去重

    #计算相似度矩阵
    def similarity(self):
        U = {}
        V = {}
        #遍历所有物品
        for item, users in self.item_users.items():
            #双重循环，计算两个用户的余弦相似度
            for u in users:
                for v in users:
                    if u != v:
                        self.similars.setdefault(u, {})
                        self.similars[u].setdefault(v, 0.)
                        U.setdefault(u, {})
                        U[u].setdefault(v, 0.)
                        V.setdefault(u, {})
                        V[u].setdefault(v, 0.)
                        #获取不同用户对相同物品的评分
                        x = self.user_ratings[u][item]
                        y = self.user_ratings[v][item]
                        self.similars[u][v] += x * y
                        U[u][v] += x * x
                        V[u][v] += y * y

        #双重循环，计算两个用户的余弦相似度
        for u, v_cnts in self.similars.items():
            for v, cnt in v_cnts.items():
                if ((U[u][v] ** 0.5) * (V[u][v] ** 0.5)) == 0:
                    self.similars[u][v] = 0
                else:
                    self.similars[u][v] = self.similars[u][v] / ((U[u][v] ** 0.5) * (V[u][v] ** 0.5))

    #计算当前用户的推荐列表
    def recommendation(self, user):
        rank = {}
        #遍历所有物品
        for item in self.total_items:
            #如果当前物品未被用户评分
            if item not in self.user_items[user]:
                count = 0
                rank.setdefault(item, 0.)
                sum_similar = 0
                #按照相似度从大到小遍历与当前用户的相似用户
                for user_v, similar in sorted(self.similars[user].items(), key = operator.itemgetter(1), reverse = True):
                    if count == self.k:
                        break
                    #如果相似用户评价了当前物品，根据相似度计算评分
                    if item in self.user_items[user_v]:
                        count += 1
                        rank[item] += similar * self.user_ratings[user_v][item]
                        sum_similar += similar
                #没有相似用户评价当前物品
                if sum_similar != 0:
                    rank[item] /= sum_similar
                else:
                    rank[item] = 0

        return rank

#计算RMSE        
def RMSE(old_ratings, new_ratings):
    res = 0
    for item in old_ratings:
        res += (old_ratings[item] - new_ratings[item]) ** 2
    res = (res / len(old_ratings)) ** 0.5
    return res

if __name__ == "__main__": 
    #读入数据   
    path = "data/ratings.csv"
    k = 30
    userBasedCF = UserBasedCF(k)
    userBasedCF.load_data(path)

    #选择一个用户，从该用户中选择n个物品
    user = '610'
    #random_items = random.sample(userBasedCF.user_items[user], 20)
    random_items = userBasedCF.user_items[user][0 : 20]

    #记录n个物品的原始评分
    old_ratings = {}
    for item in random_items:
        old_ratings.setdefault(item, 0.)
        old_ratings[item] = userBasedCF.user_ratings[user][item]
        #从数据中删除该n个物品
        userBasedCF.item_users[item].remove(user)
        userBasedCF.user_items[user].remove(item)

    #获取用户的推荐列表
    userBasedCF.similarity()
    rank = userBasedCF.recommendation(user)

    #记录n个物品的新评分
    new_ratings = {}
    for item in random_items:
        new_ratings.setdefault(item, 0.)
        new_ratings[item] = rank[item]

    #计算rmse
    rmse = RMSE(old_ratings, new_ratings)

    print('-------------user based CF-------------')
    print('user:', user, '   k:', k)
    print('rmse:', rmse)