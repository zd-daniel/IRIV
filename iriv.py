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

    def iteration(self, data, label, iter_num=100, min_dimension=10):
        self.data, self.label = data, label
        for j in range(iter_num):
            start_time = time.time()
            store_variables, remove_variables = self.__calculate_informative_variable(data, label)
            print(store_variables)
            print(remove_variables)
            print('-------------')
            if np.sum(remove_variables) == 0 or np.sum(store_variables) <= min_dimension:
                data = data[:, store_variables]
                print('The iterative rounds of IRIV have been finished, now enter into the process of backward elimination!\n')
                break

            data = data[:, store_variables]
            print('The %d th round of IRIV has finished!  ' % (j + 1))
            print('Remain %d / %d  variable, using time: %g seconds!\n' % (data.shape[1],
                                                                           self.data.shape[1],
                                                                           time.time() - start_time))

        data, _ = self.back_elimination.iteration(data, label)
        self.remain_data = data
        return data

    def __calculate_informative_variable(self, data, label):
        '''返回强弱信息变量 和无 干扰信息的索引'''
        rmsecv5, A = self.__calculate_rmsecv(data, label)
        rmsecv_origin = np.tile(rmsecv5[:, 0].reshape(-1, 1), (A.shape[1], ))
        rmsecv_replace = rmsecv5[:, 1:]

        rmsecv_exclude = rmsecv_replace.copy()
        rmsecv_include = rmsecv_replace.copy()
        rmsecv_exclude[A == 0] = rmsecv_origin[A == 0]
        rmsecv_include[A == 1] = rmsecv_origin[A == 1]
        exclude_mean = np.mean(rmsecv_exclude, axis=0)
        include_mean = np.mean(rmsecv_include, axis=0)

        p_val, DMEAN, H = [], [], []
        for i in range(A.shape[1]):
            _, pVal = stats.mannwhitneyu(rmsecv_exclude[:, i], rmsecv_include[:, i], alternative='two-sided')
            H.append(int(pVal <= 0.05))

            # # Just a trick, indicating uninformative and interfering variable if Pvalue>1
            temp_DMEAN = exclude_mean[i] - include_mean[i]
            if temp_DMEAN < 0:
                pVal = pVal + 1

            p_val.append(pVal)
            DMEAN.append(temp_DMEAN)
        p_val = np.stack(p_val)
        DMEAN = np.stack(DMEAN)
        H = np.stack(H)

        strong_inform = (H == 1) * (p_val < 1)
        weak_inform = (H == 0) * (p_val < 1)
        un_inform = (H == 0) * (p_val >= 1)
        interfering = (H == 1) * (p_val >= 1)
        remove_variables = un_inform | interfering
        store_variables = strong_inform | weak_inform
        return store_variables, remove_variables

    def __calculate_rmsecv(self, data, label):
        A, row = self.__generate_binary_matrix(data)

        rmsecv5 = np.zeros((row, data.shape[1]+1))  # PLS的五折交叉验证计算rmse
        for k, sub_a in enumerate(A):
            sub_data = data[:, sub_a == 1]

            n_component = min(np.sum(sub_a == 1), self.max_components)
            model = PLSRegression(n_components=n_component, max_iter=1000)
            score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
            rmsecv5[k, 0] = np.sqrt(-score)

        # 分别置换每一列特征的0 1， 得到B，同样方法交叉验证得到rmsecv
        for i in range(data.shape[1]):
            B = np.copy(A)
            B[:, i] = 1 - B[:, i]
            for k, sub_b in enumerate(B):
                sub_data = data[:, sub_b == 1]

                n_component = min(np.sum(sub_b == 1), self.max_components)
                model = PLSRegression(n_components=n_component, max_iter=1000)
                score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
                rmsecv5[k, i+1] = np.sqrt(-score)
        return rmsecv5, A

    def __generate_binary_matrix(self, data):
        # 新建二值矩阵A
        if data.shape[1] >= 500:
            row = 500
        elif data.shape[1] >= 300:
            row = 300
        elif data.shape[1] >= 100:
            row = 200
        elif data.shape[1] >= 50:
            row = 100
        else:
            row = 50

        A = np.ones((row, data.shape[1]))
        A[row // 2:] = 0

        while True:
            A = np.stack([np.random.permutation(sub_a) for sub_a in A.transpose()]).transpose()
            if not np.sum(np.sum(A, axis=1) == 0):
                break
        return A, row

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
        base_score = self.__get_score(data, label)
        while True:
            scores = self.__get_partial_score(data, label, delete_index)
            if base_score < scores.min():
                break
            else:
                base_score = scores.min()
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
            scores.append(np.sqrt(-sub_score))
        return np.stack(scores)

    def __get_score(self, data, label):
        n_component = min(data.shape[1], self.max_components)
        model = PLSRegression(n_components=n_component, max_iter=1000)
        score = cross_val_score(model, data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
        return np.sqrt(-score)


def calculate_retained_index(train_data, train_label):
    start_time = time.time()
    iriv_model = IRIV()
    iriv_model.iteration(train_data, train_label)
    retained_index = iriv_model.remain_index()
    end_time = time.time()
    print('计算强弱信息变量完成，耗时 %.3fs' % (end_time - start_time))
    print('保留索引[从0开始]：', retained_index)

    return retained_index


if __name__ == '__main__':
    calculate_retained_index(train_data, train_label)
