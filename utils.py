# -*- coding: utf-8 -*-
# @Time    : 2021/3/30 14:20
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : utils.py
# @Software: PyCharm


from dataloader import load_data_label
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import numpy as np
import time

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


class IRIV:
    '''Iteratively retaining informative variables (IRIV)'''
    def __init__(self, max_components=10):
        self.max_components = max_components
        self.back_elimination = Elimination(max_components)

    def iteration(self, data, label, iter_num=1000, min_dimension=6):
        self.data, self.label = data, label
        for _ in range(iter_num):
            new_data, fai_0, fai_i = self.__calculate_informative_variable(data, label)
            if new_data.shape[1] == data.shape[1] or new_data.shape[1] < min_dimension:
                data = new_data
                break
            else:
                data = new_data

        # 是否进行U检验区分强信息和弱信息
        p_val = []
        for p in range(fai_0.shape[1]):
            _, pVal = stats.mannwhitneyu(fai_0[:, p], fai_i[:, p], alternative='two-sided')
            p_val.append(pVal)
        p_val = np.stack(p_val)

        data, _ = self.back_elimination.iteration(data, label)
        self.remain_data = data
        return data

    def __calculate_informative_variable(self, data, label):
        '''返回强或弱信息变量的索引'''
        rmsecv5, A = self.__calculate_rmsecv(data, label)

        fai_0, fai_i = np.zeros(data.shape), np.zeros(data.shape)
        fai_0[A == 0] = rmsecv5[:, 1:][A == 0]
        fai_0[fai_0 == 0] = rmsecv5[:, 0].reshape(-1, 1).repeat(data.shape[1], axis=1)[fai_0 == 0]
        Mi_in = np.mean(fai_0, axis=0)
        fai_i[A == 1] = rmsecv5[:, 1:][A == 1]
        fai_i[fai_i == 0] = rmsecv5[:, 0].reshape(-1, 1).repeat(data.shape[1], axis=1)[fai_i == 0]
        Mi_out = np.mean(fai_i, axis=0)
        DMEAN = Mi_in - Mi_out
        return data[:, np.where(DMEAN < 0)[0]], fai_0, fai_i

    def __calculate_rmsecv(self, data, label):
        rmsecv5 = np.zeros((data.shape[0], data.shape[1]+1))  # PLS的五折交叉验证计算rmse
        A = np.ones(data.shape)
        A[data.shape[0] // 2:] = 0
        A = np.stack([np.random.permutation(sub_a) for sub_a in A.transpose()]).transpose()
        for k, sub_a in enumerate(A):
            sub_data = data[:, sub_a == 1]

            n_component = min(np.sum(sub_a == 1), self.max_components)
            if n_component == 0:
                break
            model = PLSRegression(n_components=n_component, max_iter=1000)
            score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
            rmsecv5[k, 0] = -score

        # 分别置换每一列特征的0 1， 得到B，同样方法交叉验证得到rmsecv
        for i in range(data.shape[1]):
            B = np.copy(A)
            B[:, i] = 1 - B[:, i]
            for k, sub_b in enumerate(B):
                sub_data = data[:, sub_b == 1]

                n_component = min(np.sum(sub_b == 1), self.max_components)
                if n_component == 0:
                    break
                model = PLSRegression(n_components=n_component, max_iter=1000)
                score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
                rmsecv5[k, i+1] = -score
        return rmsecv5, A

    def remain_index(self):
        index = []
        for i in range(self.remain_data.shape[1]):
            index.append(np.where(np.sum(self.data - self.remain_data[:, i].reshape(-1, 1), axis=0) == 0)[0][0])
        return index


class Elimination:
    def __init__(self, max_components=10):
        self.max_components = max_components

    def iteration(self, data, label):
        delete_index = []
        while True:
            base_score = self.__get_score(data, label, delete_index)
            scores = self.__get_partial_score(data, label, delete_index)
            if base_score < scores.min():
                break
            else:
                index = np.argmin(scores)
                delete_index.append(index)
        return np.delete(data, delete_index, axis=1), delete_index

    def __get_partial_score(self, data, label, delete_index):
        scores = []
        for i in range(data.shape[1]):
            if i in delete_index:
                scores.append(np.inf)
                continue
            index = delete_index.copy()
            index.append(i)
            sub_data = np.delete(data, index, axis=1)

            n_component = min(sub_data.shape[1], self.max_components)
            model = PLSRegression(n_components=n_component, max_iter=1000)
            sub_score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
            scores.append(-sub_score)
        return np.stack(scores)

    def __get_score(self, data, label, delete_index):
        if len(delete_index) > 0:
            data = np.delete(data, delete_index, axis=1)

        n_component = min(data.shape[1], self.max_components)
        model = PLSRegression(n_components=n_component, max_iter=1000)
        score = cross_val_score(model, data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
        return -score


def main():
    # 67J09140316Or
    train_path_4_in = 'data/第二次数据/五粮液检测/67J09140316Or/67JIn09140315Or.csv'
    test_path_4_in = 'data/第二次数据/五粮液检测/67J09140316Or/67JIn03160319Or.csv'
    train_path_4_out = 'data/第二次数据/五粮液检测/67J09140316Or/67JOut09140315Or.csv'
    test_path_4_out = 'data/第二次数据/五粮液检测/67J09140316Or/67JOut03160319Or.csv'

    train_data_4_in_left, _, train_label_4_in, _ = load_data_label(train_path_4_in, False)
    test_data_4_in_left, _, test_label_4_in, _ = load_data_label(test_path_4_in, False)
    train_data_4_out_left, _, train_label_4_out, _ = load_data_label(train_path_4_out, False)
    test_data_4_out_left, _, test_label_4_out, _ = load_data_label(test_path_4_out, False)

    train_data, train_label, test_data, test_label = train_data_4_out_left, \
                                                     train_label_4_out, \
                                                     test_data_4_out_left, \
                                                     test_label_4_out

    start_time = time.time()
    iriv_model = IRIV()
    iriv_model.iteration(train_data, train_label)
    retained_index = iriv_model.remain_index()
    end_time = time.time()
    try:
        print('计算强弱信息变量完成，耗时 %.3fs' % (end_time - start_time))
    except:
        pass
    print(retained_index)
    # retained_index = np.array([0, 1, 2, 4, 5, 6, 8, 12, 16, 25, 26, 29, 30, 31, 32, 33, 37, 38, 39, 45, 46, 47, 49])
    # retained_index = np.array([0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 21, 24, 25, 26, 29, 30, 31, 32, 33, 37, 38, 39, 40, 45, 46, 48, 49])

    # iriv_model = IRIV()
    # delete_index = iriv_model.iteration(train_data, train_label[:, 0])
    # train_data = np.delete(train_data, delete_index, axis=1)
    # test_data = np.delete(test_data, delete_index, axis=1)


if __name__ == '__main__':
    main()
