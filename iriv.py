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

