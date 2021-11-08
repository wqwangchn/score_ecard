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
from util import progress_bar, get_weighted_std
import pandas as pd
import numpy as np
import warnings
from score_ecard import score_card
warnings.filterwarnings('ignore')

class ECardModel():
    def __init__(self,kwargs_lr=None, kwargs_rf=None):
        self.params_rf = {
            'n_estimators': 100,
            'max_depth': 10,
            'max_features': 0.5,
            'min_samples_leaf': 0.05,
            'bootstrap': False, #全量数据集建树
            'random_state': 666
        }
        self.params_lr = {
            "class_weight": 'balanced',
            "penalty": 'l1',
            "C": 0.2,
            "solver": 'liblinear'
        }
        if kwargs_rf:
            self.params_rf.update(**kwargs_rf)
        if kwargs_lr:
            self.params_lr.update(**kwargs_lr)
        self.rf_cards = []
        self.score_model = score_card.ScoreCardModel()

    def fit(self, df_X, df_Y, validation_X:pd.DataFrame=None, validation_Y:pd.DataFrame=None, sample_weight=None):
        df_y, df_Y = self.check_label(df_Y)
        clf_rf = RandomForestClassifier(**self.params_rf)
        clf_rf.fit(df_X, df_y, sample_weight=sample_weight)
        rf_boundaries = rf_bolcks.get_randomforest_blocks(clf_rf,col_name=df_X.columns)
        gl_boundaries = self.get_boundaries_merge(rf_boundaries)
        self.blocks = gl_boundaries
        init_ecard = self.get_init_ecard(df_X,gl_boundaries)
        print("start model fitting ...")
        pre_y=np.array([0])

        # vlidation part
        validation_idx = False
        if type(validation_X) != type(None):
            assert validation_X.shape[0] == validation_Y.shape[0]
            df_valy, df_valY = self.check_label(validation_Y)
            pre_valy = np.array([0])
            validation_idx=True

        for i,tree_bins in enumerate(rf_boundaries):
            clf_lr = LogisticRegression(**self.params_lr)
            sample_idx = df_X.sample(frac=1.0, replace=True, weights=sample_weight, random_state=i).index
            df_woe = woe.get_woe_card(df_X.loc[sample_idx], df_Y.loc[sample_idx], tree_bins)
            x = self.get_woe_features(df_X, df_woe, tree_bins)
            clf_lr.fit(x.loc[sample_idx], df_y.loc[sample_idx], sample_weight=sample_weight.loc[sample_idx])
            clf_lr.col_name=x.columns
            tree_card = self.get_score_card(clf_lr,df_woe)
            self.rf_cards.append(tree_card)
            pre_y = pre_y + clf_lr.predict_proba(x)[:,1]
            cur_pre = (pre_y/(i+1))
            train_auc = self.score_model.get_auc(cur_pre,df_y, pre_target=1)[0]
            validation_info=None
            if validation_idx:
                valx = self.get_woe_features(validation_X, df_woe, tree_bins)
                pre_valy = pre_valy + clf_lr.predict_proba(valx)[:, 1]
                cur_pre = (pre_valy / (i + 1))
                val_auc = self.score_model.get_auc(cur_pre, df_valy, pre_target=1)[0]
                validation_info = "vlidation_auc={}".format(val_auc)
            print("sep_{}:\tauc={} {}".format(i+1,train_auc,validation_info))
        self.score_ecard = self.get_cards_merge(self.rf_cards, init_ecard)

    def check_label(self, df_Y):
        if df_Y.shape[-1] == 1:
            df_Y.columns = ['label']
        if 'label' not in df_Y.columns:
            df_Y['label'] = df_Y.iloc[:,0]
        df_y=df_Y.label
        return df_y, df_Y

    def get_woe_features(self, df_X,df_card,dict_blocks):
        df_x = pd.DataFrame()
        len_ = len(dict_blocks)
        for i, (_field, _bin) in enumerate(dict_blocks.items()):
            progress_bar(i, len_-1)
            _card = df_card[df_card.特征字段 == _field]
            _card.loc[:,'分箱'] = _card['分箱'].astype(str)
            _data = pd.cut(df_X[_field].fillna(0), bins=_bin)
            out = pd.DataFrame(_data).astype(str).join(_card.set_index('分箱'), on=_data.name, how='left')
            col = [i for i in out.columns if 'woe' in i]
            col2 = [_field + "_" + i for i in col]
            df_x[col2] = out[col]
        return df_x

    def get_score_card(self,clf_lr,df_woe):
        coef_dict = dict(zip(clf_lr.col_name, clf_lr.coef_[0]))
        init_score = self.score_model.score_offset
        base_score = self.score_model.score_factor * clf_lr.intercept_[0]
        woe_col = [i for i in df_woe.columns if 'woe' in i]
        get_coef_summary = lambda x: np.sum([coef_dict.get(x['特征字段'] + '_' + i) * x[i] for i in woe_col])
        df_woe['model_score'] = self.score_model.score_factor * df_woe.apply(get_coef_summary, axis=1)
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
        print("get randomforest boundaries done")
        return gl_boundaries

    def get_init_ecard(self, df_X, gl_boundaries):
        len_=len(gl_boundaries)
        print("init ecards ...")
        bins_list=[]
        for i, (col, bins) in enumerate(gl_boundaries.items()):
            progress_bar(i,len_-1)
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
        print("init ecards done")
        return df_card

    def get_importance_(self):
        df_card_info = self.score_ecard.groupby("field_").agg(
            {'size_': [("weights", lambda x: list(x))],
             'score_': [("values", lambda x: list(x))]

             }).droplevel(0, 1)
        importance_ = df_card_info.apply(lambda x: get_weighted_std(x["values"], x["weights"]), axis=1)
        return importance_.sort_values(ascending=False)

    def predict_proba(self, df_data):
        df_score = self.predict(df_data)
        df_proba =df_score.apply(lambda x:self.score_model.score_to_probability(x))
        return df_proba

    def predict(self, df_data):
        df_score = self.get_batch_score(df_data)
        return df_score

    # 单车详细评分
    def get_single_score(self, data: dict, level_threshold=None):
        score = 0
        score_detail = {}
        df_card = self.score_ecard
        for i, row in df_card.iterrows():
            field_ = row['field_']
            bins_ = row['bins_']
            score_ = row['score_']
            class_ = row['class_']
            if field_ in ['init_base_score', 'init_model_score']:
                v = 0
            else:
                v = data.get(field_)
            try:
                v = float(v)
            except:
                score = 0
                score_detail[field_] = '异常'
                break
            if not (v > -np.inf):
                v = 0
            if v in bins_:
                score = score + score_
                score_detail[field_] = class_
            else:
                pass

        # level
        level = -1
        total_meters = data.get('run_meters', 0)
        total_seconds = data.get('run_seconds', 0)
        if (level_threshold is not None) and (total_meters >= 2000000) and (total_seconds >= 10 * 3600):
            level = np.argwhere(np.sort(level_threshold + [score]) == score)[0][0]

        out = {'score': round(score), 'level': level, 'score_detail': score_detail}
        return out

    def get_batch_score(self, df_data: pd.DataFrame):
        '''
        ## 批量查询
        :param df_data:
        :return:
        '''
        df_score = pd.DataFrame()
        df_card = self.score_ecard
        for field_, bin_ in self.blocks.items():
            _card = df_card[df_card.field_ == field_]
            _card["bins_"] = _card.bins_.astype(str)
            _data = pd.cut(df_data[field_].fillna(0), bins=bin_).astype(str)
            out = pd.DataFrame(_data).join(_card.set_index('bins_'), on=_data.name, how='left')
            df_score[field_] = out['score_']
        df_score['init_base_score'] = df_card[df_card.field_ == 'init_base_score']['score_'].values[0]
        df_score['init_model_score'] = df_card[df_card.field_ == 'init_model_score']['score_'].values[0]
        score = df_score.sum(axis=1).apply(lambda x: round(x))
        return score


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
    ecard = ECardModel(kwargs_rf={'n_estimators':2})
    ecard.fit(df_X, df_Y,df_X, df_Y,sample_weight=df_Y['label']+1)
    print(ecard.get_importance_())
