# -*- coding: utf-8 -*-
# /usr/bin/env python

'''
Author: wenqiangw
Email: wqwangchn@163.com
Date: 2021/11/5 12:23
Desc:
'''

import numpy as np

def get_randomforest_blocks(clf_rf, col_name=None, topn_feat=0, valid_feature_prob=1.0):
    '''
    利用随机森林获得最优分箱的边界值列表
    :param clf_rf:
    :param col_name:
    :param topn_feat: 各决策树内重要的特征比例
    :param valid_feature_prob: 选择随机森林有效特征的比例，剔除低重要度特征，建议在0.85~1.0
    :return:
    '''
    rf_boundary = []
    rf_fields_tuple = []
    importances = clf_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    col_set = indices[importances[indices].cumsum() <= valid_feature_prob]
    for i, treei in enumerate(clf_rf.estimators_):
        n_nodes = treei.tree_.node_count
        feature_idx = treei.tree_.feature
        if col_name is not None:
            feature = np.array([(col_name[i] if i>=0 else i)for i in feature_idx])
        else:
            feature = feature_idx
        threshold = treei.tree_.threshold
        children_left = treei.tree_.children_left
        children_right = treei.tree_.children_right
        tree_boundary = {}
        for j in range(n_nodes):
            if feature_idx[j] not in col_set:
                continue
            if children_left[j] != children_right[j]:  # 获得决策树节点上的划分边界值(除叶子结点)
                if feature[j] in tree_boundary:
                    boundary = tree_boundary.get(feature[j])
                    boundary.append(threshold[j])
                else:
                    boundary = [-np.inf, np.inf]
                    boundary.append(threshold[j])
                tree_boundary.update({feature[j]: boundary})
        for k, v in tree_boundary.items():
            sv = list(set([round(i, 4) for i in v]))
            sv.sort()
            tree_boundary.update({k: sv})
        rf_boundary.append(tree_boundary)

        # extend cross feature
        step_n = topn_feat
        cross_feature_list = []
        cut_list = [0]
        while (step_n>1):
            next_list=[]
            for cur_ in cut_list:
                left_ = children_left[cur_]
                right_ = children_right[cur_]
                if left_>-1:
                    tuple_ = [feature[cur_], feature[left_]]
                    tuple_.sort()
                    cross_feature_list.append(tuple_)
                    next_list.append(left_)
                if right_>-1:
                    tuple_ = [feature[cur_], feature[right_]]
                    tuple_.sort()
                    cross_feature_list.append(tuple_)
                    next_list.append(right_)
            cut_list = next_list
            step_n-=1
        rf_fields_tuple.append(cross_feature_list)
    return rf_boundary, rf_fields_tuple
