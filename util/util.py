# coding=utf-8
# /usr/bin/env python

'''
Author: wenqiangw
Email: wenqiangw@163.com
Date: 2021-11-5 15:17
Desc:
'''

import sys
import math
import inspect
import numpy as np

# 1.进度条展示 progress_bar(1, 100)
def progress_bar(portion, total):
    """
    total 总数据大小，portion 已经传送的数据大小
    :param portion: 已经接收的数据量
    :param total: 总数据量
    :return: 接收数据完成，返回True
    """
    part = total / 50  # 1%数据的大小
    count = math.ceil(portion / part)
    sys.stdout.write('\r')
    sys.stdout.write(('[%-50s]%.2f%%' % (('>' * count), portion / total * 100)))
    sys.stdout.flush()

    if portion >= total:
        sys.stdout.write('\n')
        return True
    return False

# 2.获取变量名
def get_retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    list_name = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return list_name

# 3.计算加权方差
def get_weighted_std(values,weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return variance