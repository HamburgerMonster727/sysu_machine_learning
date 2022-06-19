import random
import operator

class ItemBasedCF:
    def __init__(self, _k):
        self.similars = {}      #物品相似度矩阵
        self.user_items = {}    #用户包含的物品
        self.user_ratings = {}  #用户的评分
        self.total_items = []   #总共的物品列表
        self.k = _k

    def load_data(self, file_path):
        with open(file_path, "r") as f:
            for i, line in enumerate(f, 0):
                if i != 0:  
                    line = line.strip('\n')
                    user, item, rating, timestamp = line.split(',')
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
        #遍历所有用户
        for user, items in self.user_items.items():
            #双重循环，计算两个物品的余弦相似度
            for i in items:
                for j in items:
                    if i != j:
                        self.similars.setdefault(i, {})
                        self.similars[i].setdefault(j, 0.)
                        U.setdefault(i, {})
                        U[i].setdefault(j, 0.)
                        V.setdefault(i, {})
                        V[i].setdefault(j, 0.)
                        #获取相同用户对不同物品的评分
                        x = self.user_ratings[user][i]
                        y = self.user_ratings[user][j]
                        self.similars[i][j] += x * y
                        U[i][j] += x * x
                        V[i][j] += y * y

        #双重循环，计算两个物品的余弦相似度
        for i, j_cnt in self.similars.items():
            for j, cnt in j_cnt.items():
                if ((U[i][j] ** 0.5) * (V[i][j] ** 0.5)) == 0:
                    self.similars[i][j] = 0
                else:
                    self.similars[i][j] = self.similars[i][j] / ((U[i][j] ** 0.5) * (V[i][j] ** 0.5))
                    
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
                #按照相似度从大到小遍历与当前物品的相似物品
                for item_j, similar in sorted(self.similars[item].items(), key = operator.itemgetter(1), reverse = True):
                    if count == self.k:
                        break
                    #如果当前用户评价了相似物品，根据相似度计算评分
                    if item_j in self.user_items[user]:
                        count += 1
                        rank[item] += similar * self.user_ratings[user][item_j]
                        sum_similar += similar
                #当前用户没有评价相似物品
                if sum_similar != 0:
                    rank[item] /= sum_similar
                #当前用户评价的相似物品数量不足k个
                if count < self.k:
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
    k = 60
    itemBasedCF = ItemBasedCF(k)
    itemBasedCF.load_data(path)

    #随机选择一个用户，从该用户中随机选择n个物品
    user = '610'
    #random_items = random.sample(itemBasedCF.user_items[user], 20)
    random_items = itemBasedCF.user_items[user][0 : 20]

    #记录n个物品的原始评分
    old_ratings = {}
    for item in random_items:
        old_ratings.setdefault(item, 0.)
        old_ratings[item] = itemBasedCF.user_ratings[user][item]
        #从数据中删除该n个物品
        itemBasedCF.user_items[user].remove(item)

    #获取用户的推荐列表
    itemBasedCF.similarity()
    rank = itemBasedCF.recommendation(user)

    #记录n个物品的新评分
    new_ratings = {}
    for item in random_items:
        new_ratings.setdefault(item, 0.)
        new_ratings[item] = rank[item]

    #计算rmse
    rmse = RMSE(old_ratings, new_ratings)

    print('-------------item based CF-------------')
    print('user:', user, '   k:', k)
    print('rmse:', rmse)