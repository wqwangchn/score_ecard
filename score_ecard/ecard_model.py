# -*- coding: utf-8 -*-
# /usr/bin/env python

'''
Author: wenqiangw
Email: wqwangchn@163.com
Date: 2021/11/5 21:51
Desc:
'''
from interval import Interval
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from score_ecard.features import randomforest_blocks as rf_bolcks
from score_ecard.features import layered_woe as woe
from util import progress_bar
import pandas as pd
import numpy as np
from score_ecard import score_card


class ECardModel():
    def __init__(self,**kwargs):
        self.params_rf = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_leaf': 0.05,
            'bootstrap': True,
            'random_state': 666
        }
        self.params_lr = {
            "class_weight": 'balanced',
            "penalty": 'l1',
            "C": 0.2,
            "solver": 'liblinear'
        }
        self.params_rf.update(**kwargs)
        self.params_lr.update(**kwargs)
        self.rf_cards = []

    def fit(self, df_X, df_Y, sample_weight=None):
        df_y, df_Y = self.check_label(df_Y)
        clf_rf = RandomForestClassifier(**self.params_rf)
        clf_rf.fit(df_X, df_y, sample_weight=None)
        rf_boundaries = rf_bolcks.get_randomforest_blocks(clf_rf,col_name=df_X.columns)

        gl_boundaries = self.get_boundaries_merge(rf_boundaries)
        init_ecard = self.get_init_ecard(df_X,gl_boundaries)

        score_model = score_card.ScoreCardModel()
        for tree_bins in rf_boundaries:
            clf_lr = LogisticRegression(**self.params_lr)
            df_woe = woe.get_woe_card(df_X, df_Y, tree_bins)
            x = self.get_woe_features(df_X, df_woe, tree_bins)
            clf_lr.fit(x, df_y, sample_weight=sample_weight)
            clf_lr.col_name=x.columns
            tree_card = self.get_score_card(score_model,clf_lr,df_woe)
            self.rf_cards.append(tree_card)
        self.ecard = self.get_cards_merge(self.rf_cards, init_ecard)
        print(self.ecard)


    def check_label(self, df_Y):
        if df_Y.shape[-1] == 1:
            df_Y.columns = ['label']
        if 'label' not in df_Y.columns:
            df_Y['label'] = df_Y.iloc[:,0]
        df_y=df_Y[['label']]
        return df_y, df_Y

    def get_woe_features(self, df_X,df_card,dict_blocks):
        df_x = pd.DataFrame()
        print("\nget woe features ......")
        len_ = len(dict_blocks)
        for i, (_field, _bin) in enumerate(dict_blocks.items()):
            progress_bar(i, len_)
            _card = df_card[df_card.特征字段 == _field]
            _card['分箱'] = _card['分箱'].astype(str)
            _data = pd.cut(df_X[_field].fillna(0), bins=_bin)
            out = pd.DataFrame(_data).astype(str).join(_card.set_index('分箱'), on=_data.name, how='left')
            col = [i for i in out.columns if 'woe' in i]
            col2 = [_field + "_" + i for i in col]
            df_x[col2] = out[col]
        return df_x

    def get_score_card(self,score_model,clf_lr,df_woe):
        coef_dict = dict(zip(clf_lr.col_name, clf_lr.coef_[0]))
        init_score = score_model.score_offset
        base_score = score_model.score_factor * clf_lr.intercept_[0]
        woe_col = [i for i in df_woe.columns if 'woe' in i]
        get_coef_summary = lambda x: np.sum([coef_dict.get(x['特征字段'] + '_' + i) * x[i] for i in woe_col])
        df_woe['model_score'] = score_model.score_factor * df_woe.apply(get_coef_summary, axis=1)
        df_field_card = df_woe[['特征字段', '分箱', '车辆总数', 'model_score']]
        df_field_card.columns = ['field_', 'bins_', 'size_', 'score_']
        df_base_card = pd.DataFrame(
            [['init_base_score', Interval(-np.inf, np.inf, closed='right'), None, init_score],
             ['init_model_score', Interval(-np.inf, np.inf, closed='right'), None, base_score]
             ], columns=['field_', 'bins_', 'size_', 'score_'])
        df_card = df_field_card.append(df_base_card, ignore_index=True)
        return df_card

    def get_cards_merge(self, rf_cards, init_ecard):
        df_ecard=init_ecard.copy()
        df_ecard['score_']=0
        df_ecard['bins_']=df_ecard['bins_'].astype(str)
        df_ecard = df_ecard.groupby(['field_','bins_','size_'])['score_'].sum()
        len_ = len(rf_cards)
        for i,tree_card in enumerate(rf_cards):
            progress_bar(i,len_)
            icard = init_ecard.join(tree_card.set_index('field_')[['bins_', 'score_']], on='field_', how='left',
                                    rsuffix='_')
            idx = icard.apply(
                lambda x: True if str(x['bins__']) != 'nan' and (x['bins_'].overlaps(x['bins__'])) else False, axis=1)
            icard.loc[~idx, 'score_'] = 0
            icard['bins_']=icard['bins_'].astype(str)
            df_icard = icard.groupby(['field_', 'bins_', 'size_'])['score_'].sum()
            df_ecard+=df_icard/len_
        return df_ecard.reset_index()

    def get_boundaries_merge(self, boundaries_list):
        gl_boundaries={}
        for boundaries in boundaries_list:
            for k,v in boundaries.items():
                if k in gl_boundaries:
                    lv = gl_boundaries.get(k)
                    cv = list(set(v+lv))
                    cv.sort()
                    gl_boundaries.update({k:cv})
                else:
                    gl_boundaries.update({k: v})
        return gl_boundaries

    def get_init_ecard(self, df_X, gl_boundaries):
        len_=len(gl_boundaries)
        print("init df_ecards ...")
        bins_list=[]
        for i, (col, bins) in enumerate(gl_boundaries.items()):
            progress_bar(i,len_)
            data_bin = pd.cut(df_X.loc[:, col].fillna(0), bins=bins).value_counts()
            df_bin = pd.DataFrame(data_bin).sort_index().reset_index()
            df_bin.columns = ['bins_','size_']
            df_bin.insert(loc=0, column='field_', value=col)
            bins_list.append(df_bin)
        df_init_ecard = pd.concat(bins_list).reset_index(drop=True)
        df_base_card = pd.DataFrame(
            [['init_base_score', Interval(-np.inf, np.inf, closed='right'), -1],
             ['init_model_score', Interval(-np.inf, np.inf, closed='right'), -1]
             ], columns=['field_', 'bins_', 'size_'])
        df_card = df_init_ecard.append(df_base_card, ignore_index=True)
        return df_card



if __name__ == '__main__':
    df_valid = pd.read_csv("data/train_test_data.csv")
    df_train_data = df_valid[df_valid['train_test_tag'] == '训练集']
    df_test_data = df_valid[df_valid['train_test_tag'] == '测试集']
    features_col = ['trip_avg_meters',
                    'trip_avg_seconds', 'trip_avg_distance', 'high_meters_ratio',
                    'province_meters_ratio', 'high_trip_cnt_ratio',
                    'province_trip_cnt_ratio', 'curvature_g2_trip_meters_ratio',
                    'ng_23_6_seconds_ratio', 'ng_23_6_trip_cnt_ratio', 'daily_run_kmeters',
                    'daily_run_hours', 'daily_trip_cnt', 'daily_nohigh_kmeters',
                    'daily_ng_23_6_hours', 'trip_long_cnt_ratio', 'day_high_meters_ratio',
                    'ng_province_meters_ratio', 'morn_6_10_seconds_ratio',
                    'dusk_17_20_seconds_ratio', 'ng_23_6_avg_speed', 'morn_6_10_avg_speed',
                    'dusk_17_20_avg_speed', 'low_speed_seconds_ratio',
                    'low_speed_block_cnt_ratio', 'week_1_5_seconds_ratio',
                    'geohash4_top10_meters_ratio', 'trip_r30m_cnt_ratio',
                    'common_line_top30_cnt_ratio', 'ratio_meitan', 'ratio_gangtie', 'ratio_shashi',
                    'ratio_kuaidi', 'ratio_nonglinmufu', 'ratio_jiadian', 'ratio_jixie',
                    'ratio_qiche', 'ratio_fengdian', 'ratio_other'
                    ]
    df_X = df_train_data[features_col]
    df_Y = df_train_data[['label', 'label_ordinary',
                          'label_serious', 'label_major', 'label_devastating', 'label_8w']]
    ecard = ECardModel()
    ecard.fit(df_X, df_Y)