# -*- coding: utf-8 -*-
# /usr/bin/env python

'''
Author: wenqiangw
Email: wqwangchn@163.com
Date: 2021/11/6 00:11
Desc:
'''


import numpy as np
import pandas as pd

## 评分体系
class ScoreCardModel:
    def __init__(self):
        self.load_score_alpha()

    def load_score_alpha(self, odds=1 / 2, bScore=600, addScore=-30):
        '''
            bScore = offset - factor*ln(odds)
            bScore + addScore = offset - factor*ln(2*odds)
        :param odds: 坏好比
        :param bScore: 基础score分
        :param addScore: 评分变动值
        :return: bScore = offset - factor*ln(odds)
        '''

        self.score_factor = addScore / np.log(2)
        self.score_offset = bScore - addScore * np.log(odds) / np.log(2)

    def probability_to_score(self, bad_prob, eps=1e-5):
        '''
            逾期概率转换为评分
        :param bad_prob: 逾期概率
        :return: bScore = offset - factor*ln(odds)
        '''

        odds = max(bad_prob, eps) / max(1 - bad_prob, eps)
        score = self.score_offset + self.score_factor * np.log(odds)
        return int(round(score))

    def score_to_probability(self, score, eps=1e-5):
        '''
            评分转换为逾期概率
        :param score: 信用评分
        :return: bScore = offset - factor*ln(odds) -->bad_prob
        '''
        score_factor = eps if self.score_factor==0 else self.score_factor
        odds = np.e**((score - self.score_offset)/(score_factor))
        odds = -1+eps if odds == -1 else odds
        bad_prob = 1.00*odds/(1+odds)
        return bad_prob

    @classmethod
    def get_auc(cls, df_pre, df_label, pre_target=1):
        '''
        功能: 计算KS值，输出对应分割点和累计分布函数曲线图
        :param df_pre: 一维数组或series，代表模型得分
        :param df_label: 一维数组或series，代表真实的标签{0,1}
        :return: 'auc': auc值，'crossdens': TPR&FPR
        '''
        df_label = df_label.apply(lambda x: 1 if x == pre_target else 0)
        crossfreq = pd.crosstab(df_pre, df_label).sort_index(ascending=False)  # 按照预测概率从大到小排序作为阈值
        crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
        crossdens.columns = ['fpr', 'tpr']
        crossdens.name = 'pre_threshold'
        fpr = crossdens.loc[:, 'fpr']
        tpr = crossdens.loc[:, 'tpr']
        auc = np.trapz(y=tpr, x=fpr)
        return auc, crossdens

    @classmethod
    def get_g7_auc(cls, df_pre, df_got_fee, df_report_fee):
        '''
        功能: 计算auc值，输出对应分割点和累计分布函数曲线图
        :param df_pre: 一维数组或series，代表模型得分
        :param df_got_fee: 一维数组或series，代表已赚保费
        :param df_report_fee: 一维数组或series，代表赔付额
        :return: 'auc': auc值，'crossdens': TPR&FPR
        '''
        df_data = pd.concat(
            [df_pre.reset_index(drop=True), df_got_fee.reset_index(drop=True), df_report_fee.reset_index(drop=True)],
            axis=1, ignore_index=True)
        df_data.columns = ['pre', 'got_fee', 'report_fee']
        df_data = df_data.set_index('pre').sort_index(ascending=False)  # 按照预测概率从大到小排序作为阈值
        crossdens = df_data.cumsum(axis=0) / df_data.sum()
        crossdens.columns = ['fpr', 'tpr']
        crossdens.name = 'pre_threshold'
        fpr = crossdens.loc[:, 'fpr']
        tpr = crossdens.loc[:, 'tpr']
        auc = np.trapz(y=tpr, x=fpr)
        return auc, crossdens

    @classmethod
    def get_ks(cls, df_pre, df_label, pre_target=1):
        '''
        功能: 计算KS值，输出对应分割点和累计分布函数曲线图
        :param df_pre: 一维数组或series，代表模型得分
        :param df_label: 一维数组或series，代表真实的标签
        :return:
            'ks': KS值
            'crossdens': 好坏客户累积概率分布以及其差值gap
        '''
        df_label = df_label.apply(lambda x: 1 if x == pre_target else 0)
        crossfreq = pd.crosstab(df_pre, df_label).sort_index(ascending=False)  # 按概率降序排序
        crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
        crossdens.columns = ['fpr', 'tpr']
        crossdens.name = 'pre_threshold'
        crossdens['gap'] = abs(crossdens['fpr'] - crossdens['tpr'])
        ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
        return ks, crossdens

    @classmethod
    def get_g7_ks(cls, df_pre, df_got_fee, df_report_fee):
        '''
        功能: 计算auc值，输出对应分割点和累计分布函数曲线图
        :param df_pre: 一维数组或series，代表模型得分
        :param df_got_fee: 一维数组或series，代表已赚保费
        :param df_report_fee: 一维数组或series，代表赔付额
        :return: 'auc': auc值，'crossdens': TPR&FPR
        '''
        df_data = pd.concat(
            [df_pre.reset_index(drop=True), df_got_fee.reset_index(drop=True), df_report_fee.reset_index(drop=True)],
            axis=1)
        df_data.columns = ['pre', 'got_fee', 'report_fee']
        df_data = df_data.set_index('pre').sort_index(ascending=False)  # 按照预测概率从大到小排序作为阈值
        crossdens = df_data.cumsum(axis=0) / df_data.sum()
        crossdens.columns = ['fpr', 'tpr']
        crossdens.name = 'pre_threshold'
        crossdens['gap'] = abs(crossdens['fpr'] - crossdens['tpr'])
        ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
        return ks, crossdens

# 评分校准
def get_calibration_score(raw_score):
    pass

if __name__ == '__main__':
    model_score = ScoreCardModel()
    score = model_score.probability_to_score(0.5)
    print(score)