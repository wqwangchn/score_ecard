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
from sklearn.metrics import classification_report
from score_ecard.features import randomforest_blocks as rf_bolcks
from score_ecard.features import layered_woe as woe
from util import progress_bar, get_weighted_std
import pandas as pd
import numpy as np
import warnings
from score_ecard import score_card
warnings.filterwarnings('ignore')

class ECardModel():
    def __init__(self,kwargs_lr=None, kwargs_rf=None, sample_frac=1.0, cross_hierarchy=0, is_best_bagging=False):
        '''
        :param kwargs_lr:
        :param kwargs_rf:
        :param sample_frac: 样本再抽样比例，样本抽样比例 = 0.6*sample_frac
        :param cross_hierarchy: 特征交叉层级，可参考参数设置为[2，3]，默认不进行特征扩展即cross_hierarchy=0
        :param is_best_bagging:
        '''
        self.params_rf = {
            'n_estimators': 100,
            'max_depth': 5,
            'max_features': 'auto',
            'min_samples_leaf': 0.01,
            'bootstrap': True,
            'class_weight': "balanced",
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
        self.sample_frac = sample_frac
        self.cross_hierarchy = cross_hierarchy
        self.is_best_bagging = is_best_bagging
        self.score_model = score_card.ScoreCardModel()

    def fit(self, df_X, df_Y, validation_X:pd.DataFrame=None, validation_Y:pd.DataFrame=None, sample_weight=None):
        df_Y, df_y, df_y_binary= self.check_label(df_Y)
        if type(sample_weight)==type(None):
            sample_weight=df_y*0+1
        validation_idx = False
        if type(validation_X) != type(None):
            assert validation_X.shape[0] == validation_Y.shape[0]
            df_valY, df_valy, df_valy_binary = self.check_label(validation_Y)
            pre_valy = np.array([0])
            validation_idx = True

        pre_y = np.array([0])
        clf_rf = RandomForestClassifier(**self.params_rf)
        clf_rf.fit(df_X, df_y, sample_weight=sample_weight)
        rf_report = classification_report(df_y, clf_rf.predict(df_X))
        print('RF-report:','\n',rf_report)

        rf_boundaries, _ = rf_bolcks.get_randomforest_blocks(clf_rf, col_name=df_X.columns)
        if self.cross_hierarchy>1:
            ext_boundaries = self.calc_cross_boundaries(df_X,df_y,sample_weight)
        else:
            ext_boundaries = {}
        if len(ext_boundaries)>0:
            df_X = self.get_cross_features(df_X)
            validation_X = self.get_cross_features(validation_X)

        init_ecard = self.get_init_ecard(df_X, rf_boundaries, ext_boundaries)
        print("start model fitting ...")
        best_auc = -999
        best_insurance_auc = -999
        for i,tree_bins in enumerate(rf_boundaries):
            for k,v in ext_boundaries.items():
                idx_ = True
                for jj in k[4:].split("-"):
                    if jj not in tree_bins.keys():
                        idx_ = False
                        break
                if idx_:
                    tree_bins.update({k:v})
            sample_idx = df_X.sample(
                frac=1.0, replace=True, weights=sample_weight, random_state=i
            ).sample(frac=self.sample_frac, replace=False, random_state=i).index
            df_woe = woe.get_woe_card(df_X.loc[sample_idx], df_Y.loc[sample_idx], tree_bins)
            x = self.get_woe_features(df_X, df_woe, tree_bins)
            clf_lr = LogisticRegression(**self.params_lr)
            clf_lr.fit(x.loc[sample_idx], df_y_binary.loc[sample_idx], sample_weight=sample_weight.loc[sample_idx])
            clf_lr.col_name=x.columns
            tree_card = self.get_score_card(clf_lr,df_woe)
            pre_y = pre_y + clf_lr.predict_proba(x)[:,1:].sum(axis=1)
            cur_pre = (pre_y/(i+1))
            train_auc = self.score_model.get_auc(cur_pre,df_y_binary, pre_target=1)[0]
            train_info = "train_auc={}".format(round(train_auc, 4))
            if self.is_best_bagging and (train_auc < best_auc):
                vote = False
            else:
                best_auc = train_auc
                vote = True
            if ('fee_got' in df_Y.columns) and ('report_fee' in df_Y.columns):
                train_insurance_auc = \
                    self.score_model.get_g7_auc(pd.DataFrame(cur_pre), df_Y["fee_got"], df_Y["report_fee"], )[0]
                train_info = "{}, train_insurance_auc={}".format(train_info, round(train_insurance_auc, 4))
                if train_insurance_auc >= best_insurance_auc:
                    best_insurance_auc = train_insurance_auc
                    vote = True
            if vote:
                self.rf_cards.append(tree_card)
            else:
                pre_y = pre_y - clf_lr.predict_proba(x)[:,1:].sum(axis=1)

            validation_info=None
            if validation_idx and vote:
                valx = self.get_woe_features(validation_X, df_woe, tree_bins)
                pre_valy = pre_valy + clf_lr.predict_proba(valx)[:, 1:].sum(axis=1)
                cur_pre = (pre_valy / (i + 1))
                validation_auc = self.score_model.get_auc(cur_pre, df_valy_binary, pre_target=1)[0]
                validation_info = "validation_auc={}".format(round(validation_auc, 4))
                if ('fee_got' in df_valY.columns) and ('report_fee' in df_valY.columns):
                    validation_insurance_auc = self.score_model.get_g7_auc(pd.DataFrame(cur_pre), df_valY["fee_got"], df_valY["report_fee"], )[0]
                    validation_info = "{}, validation_insurance_auc={}".format(validation_info,round(validation_insurance_auc,4))
            print("sep_{}:\t{}\t{}".format(i+1,train_info,validation_info))
        self.score_ecard = self.get_cards_merge(self.rf_cards, init_ecard)

    def check_label(self, df_label):
        df_Y=df_label.copy()
        if df_Y.shape[-1] == 1:
            df_Y.columns = ['label']
        if 'label' not in df_Y.columns:
            df_Y['label'] = df_Y.iloc[:,0]
        df_y=df_Y.label.astype(int)
        if df_y.unique().size>2:
            df_y_binary = df_y.apply(lambda x: 1 if x>0 else 0)
        else:
            df_y_binary = df_y
        df_Y['label']=df_y_binary
        ## test---
        if 'label_binary' in df_Y.columns:
            df_y_binary = df_Y["label_binary"]
        ## ---end
        return df_Y, df_y, df_y_binary

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
        df_ecard['bins_str']=df_ecard['bins_'].astype(str)
        df_init_ecard = df_ecard.groupby(['field_','bins_str','size_'])['bins_'].max()
        df_ecard = df_ecard.groupby(['field_','bins_str','size_'])['score_'].sum()
        len_ = len(rf_cards)
        for i,tree_card in enumerate(rf_cards):
            progress_bar(i,len_)
            icard = init_ecard.join(tree_card.set_index('field_')[['bins_', 'score_']], on='field_', how='left',
                                    rsuffix='_')
            idx = icard.apply(
                lambda x: True if str(x['bins__']) != 'nan' and (x['bins_'].overlaps(x['bins__'])) else False, axis=1)
            icard.loc[~idx, 'score_'] = 0
            icard['bins_str']=icard['bins_'].astype(str)
            df_icard = icard.groupby(['field_', 'bins_str', 'size_'])['score_'].sum()
            df_ecard+=df_icard/len_
        df_ecard = df_ecard.reset_index()
        df_ecard['bins_'] = df_init_ecard['bins_']
        return df_ecard.reset_index()

    def get_boundaries_merge(self, boundaries_list,boundaries_ext={}):
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
        for k, v in boundaries_ext.items():
            if k in gl_boundaries:
                lv = gl_boundaries.get(k)
                cv = list(set(v + lv))
                cv.sort()
                gl_boundaries.update({k: cv})
            else:
                gl_boundaries.update({k: v})
        return gl_boundaries

    def get_init_ecard(self, df_X, rf_boundaries, ext_boundaries):
        gl_boundaries = self.get_boundaries_merge(rf_boundaries, ext_boundaries)
        self.blocks = gl_boundaries
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

    def get_ext_features(self, data):
        if hasattr(self, 'feature_ext_info'):
            df_ext = self.feature_ext_info.get("df_ext")
            ext_columns = [i for i in df_ext.columns if 'ext-' in i]
            for i in ext_columns:
                colinfo = str(i).split('-')
                if 'ext' == colinfo[0]:
                    idx_equ = True
                    for icol in colinfo[1:]:
                        if icol not in df_ext.columns:
                            break
                        idx_equ = idx_equ & df_ext[icol].apply(lambda x: data.get(icol) in x)
                    data.update({i: df_ext.loc[idx_equ, i].values[0]})
        return data

    # 单车详细评分
    def get_single_score(self, data: dict, level_threshold=None):
        data = self.get_ext_features(data)
        score = 0
        score_detail = {}
        df_card = self.score_ecard
        for i, row in df_card.iterrows():
            field_ = row['field_']
            bins_ = row['bins_']
            score_ = row['score_']
            # class_ = row['class_']
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
                # score_detail[field_] = class_
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
        if self.cross_hierarchy>1:
            df_data = self.get_cross_features(df_data)
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

    def calc_cross_boundaries(self, df_X, df_y, sample_weight):
        clf_tree = RandomForestClassifier(
            n_estimators=1,
            max_depth=20,
            max_features=None,
            bootstrap=False,
            min_samples_leaf=0.01,
            random_state=666,
        )
        clf_tree.fit(df_X, df_y, sample_weight=sample_weight)
        _, rf_cross_tuple = rf_bolcks.get_randomforest_blocks(clf_tree, col_name=df_X.columns,
                                                              topn_feat=self.cross_hierarchy)

        boundaries = {}
        for col_ in np.unique(rf_cross_tuple):
            clf_tree = RandomForestClassifier(
                n_estimators=1,
                max_depth=2,
                max_features=None,
                bootstrap=False,
                min_samples_leaf=0.01,
                random_state=666,
            )
            clf_tree.fit(df_X[[col_]], df_y)
            col_boundaries, _ = rf_bolcks.get_randomforest_blocks(clf_tree, col_name=[col_])
            col_boundaries = self.get_boundaries_merge(col_boundaries)
            boundaries.update(col_boundaries)

        df_X_ext = pd.DataFrame()
        for k,bv in boundaries.items():
            df_X_ext[k]=pd.cut(df_X[k],bins=bv)
        df_X_ext.drop_duplicates(inplace=True)
        df_X_ext.reset_index(drop=True,inplace=True)

        ext_boundaries = {}
        for i,j in [i for item in rf_cross_tuple for i in item]:
            if i==j:
                continue
            col_='ext-{}-{}'.format(i,j)
            if col_ in ext_boundaries:
                continue
            df_tmp = pd.DataFrame(df_X_ext[[i,j]].astype(str).apply(lambda x: '-'.join(x), axis=1).drop_duplicates())
            df_tmp.columns=['tmp_']
            df_tmp[col_] = range(len(df_tmp))
            if len(df_tmp)<2:
                boundaries_i = [-np.inf,np.inf]
            else:
                boundaries_i = np.array(range(len(df_tmp)))
                boundaries_i = list((boundaries_i[:-1]+boundaries_i[1:])/2)
                boundaries_i[0] = -np.inf
                boundaries_i[-1] = np.inf
            ext_boundaries.update({col_:boundaries_i})
            df_X_ext['tmp_'] = df_X_ext[[i,j]].astype(str).apply(lambda x: '-'.join(x), axis=1)
            df_X_ext = df_X_ext.join(df_tmp.set_index('tmp_'),on='tmp_',how='inner')
        if 'tmp_' in df_X_ext.columns:
            del df_X_ext['tmp_']
        self.feature_ext_info = {'df_ext':df_X_ext,'boundaries_ext':boundaries}
        return ext_boundaries

    def get_cross_features(self, df_data):
        df_X_ext = self.feature_ext_info.get('df_ext')
        boundaries_ext = self.feature_ext_info.get('boundaries_ext')
        df_X = df_data.copy()
        df_tmp = pd.DataFrame()
        for k,bv in boundaries_ext.items():
            df_tmp[k]=pd.cut(df_X[k],bins=bv).astype(str)
        for icol in df_X_ext.columns:
            if 'ext-' not in icol:
                continue
            col_ = icol[4:].split('-')
            df_tmp['tmp_'] = df_tmp[col_].apply(lambda x: '-'.join(x), axis=1)
            df_X_ext['tmp_'] = df_X_ext[col_].astype(str).apply(lambda x: '-'.join(x), axis=1)
            df_ext_tmp = df_X_ext[['tmp_',icol]].drop_duplicates()
            df_tval = df_tmp.join(df_ext_tmp.set_index('tmp_'),on='tmp_',how='inner')[icol]
            assert len(df_tval)==len(df_X), (len(df_tval),len(df_X))
            df_X[icol] = df_tval
        return df_X



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
                          'label_serious', 'label_major', 'label_devastating', 'label_8w','fee_got','report_fee']]
    df_Y['label']=df_Y.apply(lambda x:x['label'] if x["report_fee"]<5000 else 2,axis=1)
    ecard = ECardModel(kwargs_rf={'n_estimators':2},cross_hierarchy=3, is_best_bagging=True)
    ecard.fit(df_X, df_Y,df_X, df_Y,sample_weight=df_Y['label']+1)
    print(ecard.get_importance_())
    print(ecard.predict(df_X))
    data = {
        'carnum': '皖SF8698',
         'trip_cnt': 90.0,
         'run_meters': 33469500.089999996,
         'run_seconds': 866187.0,
         'trip_avg_meters': 371883.3343333333,
         'trip_avg_seconds': 9624.3,
         'trip_avg_distance': 193177.38888888888,
         'high_meters_ratio': 0.8559960505224269,
         'province_meters_ratio': 0.014080034620558924,
         'high_trip_cnt_ratio': 0.7666666666666667,
         'province_trip_cnt_ratio': 0.03333333333333333,
         'curvature_g2_trip_meters_ratio': 0.3860073136813918,
         'ng_23_6_seconds_ratio': 0.0823170978091336,
         'ng_23_6_trip_cnt_ratio': 0.14444444444444443,
         'daily_run_kmeters': 0.14444444444444443,
         'daily_run_hours': 5.423234261889526,
         'daily_trip_cnt': 2.0285780101204547,
         'daily_nohigh_kmeters': 108.63576649126179,
         'daily_ng_23_6_hours': 0.44642490517780453,
         'trip_long_cnt_ratio': 0.2222222222222222,
         'day_high_meters_ratio': 0.4270326656080033,
         'ng_province_meters_ratio': 0.004463795383804909,
         'morn_6_10_seconds_ratio': 0.22645225569074576,
         'dusk_17_20_seconds_ratio': 0.26537572140888743,
         'ng_23_6_avg_speed': 136.96548225856216,
         'morn_6_10_avg_speed': 119.05333979097628,
         'dusk_17_20_avg_speed': 143.4565817501577,
         'low_speed_seconds_ratio': 0.12923537296218945,
         'low_speed_block_cnt_ratio': 0.06950028719126938,
         'week_1_5_seconds_ratio': 0.2588990599027693,
         'geohash4_top10_meters_ratio': 0.32584831893735033,
         'trip_r30m_cnt_ratio': 0.7211538461538461,
         'common_line_top30_cnt_ratio': 0.1,
         'triggertime': '2021-04-04 14:49:42',
         'triggertime_end': '2021-06-27  00:16:22',
         'ratio_meitan': 0.004785206867784571,
         'ratio_gangtie': 0.07997876021761313,
         'ratio_nonglinmufu': 0.5200959760919361,
         'ratio_shashi': 0.13649491967093144,
         'ratio_kuaidi': 0.09816298966354879,
         'ratio_jiadian': 0.045777212715108415,
         'ratio_fengdian': 0.0,
         'ratio_other': 0.0,
         'ratio_jixie': 0.11433135879828957,
         'ratio_qiche': 0.0003735759747879492,
         'card_score': 631,
         'card_level': 1,
         'hangye': '农林',
         'model_version': 'card_V2.1',
         'tag': '白中白',
         'source': 10.0,
         'score': 0.0,
         'truckno': 0,
         'pred': 0,
         'if_liuan': 0,
         'anhui_mileage_province_rate': 0.17674700919024314,
         'jiangsu_mileage_province_rate': 0.027259005588571722,
         'guangdong_mileage_province_rate': 0.0,
         'montainu_rate': 0.00022354681067481965,
         'car_got': 1.0,
         'report_num': 0,
         'report_fee': 0.0,
         'train_test_split_rand_value': 0.0512,
         'train_test_idx': 'train',
         'fee_got': 15000.0,
         'label': 0,
         'label_ordinary': 0,
         'label_serious': 0,
         'label_major': 0,
         'label_devastating': 0,
         'label_10w': 0,
         'label_8w': 0,
         'label_5w': 0,
         'lr_pre': 0.327752870486023,
         'model1_score': 631
        }
    print(ecard.get_single_score(data=data, level_threshold=None))
