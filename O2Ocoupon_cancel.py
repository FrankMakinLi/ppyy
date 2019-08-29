# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:37:22 2019

@author: lisong
"""
这个作业用来完成天池网站上的O2O优惠券使用预测。
这个项目是个新手练手项目。主要目的是根据模型找到核销优惠券的特征。
这其实是一个二分类任务，也就是基于特征输出是和否。
方便广告投入回收。
本赛题目标是预测投放的优惠券是否核销。针对此任务及一些相关背景知识，使用优惠券核销
预测的平均AUC（ROC曲线下面积）作为评价标准。 即对每个优惠券coupon_id单独计算核销
预测的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。 关于AUC的含义与具
体计算方法，可参考维基百科。
1.数据集准备
    1.1数据集下载
        已下载。
    1.2数据集整理。
        根据当前获得的数据集，train数据集，特征并不多，仅有7个，有outlier和nan数据集
        需要关注，还有时间序列数据集其格式不正确，不利于计算。因为特征较少，所以应该
        增加一些特征，比如事件序列的interal化，方便统计和计算频率。
        数据集较大。占用内存较多。官方提供了jupyter book类似的环境，而且提供了较好的计算资源
        但如何调用云端文件，目前还不清楚。看起来应该是linux环境以及shell命令。
        如何使用需要找相关的文档。强迫自己学jupyter book。
    1.3适配模型。
        希望通过多种模型的尝试，也就是之前学的各种机器学习的模型。以及sklearn的学习。
        
    1.4AUC值计算
        根据Sklearn的相关模块，可以直接计算，最后选择结果最好的一次上传。
        

import pandas as pd
import numpy as np
import math
import time
ccf_offline_test=pd.read_csv(r'C:\Users\LISONG\Desktop\数据分析档案\coupon_cancel\ccf_offline_stage1_test_revised.csv')
coupon_sample = pd.read_csv(r'C:\Users\LISONG\Desktop\数据分析档案\coupon_cancel\sample_submission.csv',header=0)
ccf_offline_train=pd.read_csv(r'C:\Users\LISONG\Desktop\数据分析档案\coupon_cancel\ccf_offline_stage1_train.csv')
"""
shape是(1754884, 7)，明显比online要少很多。相比online，唯一的区别是distance和action
offline有distance，online有action
占用内存还可以接受
ccf_offline_train.info(null_counts=1)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1754884 entries, 0 to 1754883
Data columns (total 7 columns):
User_id          1754884 non-null int64
Merchant_id      1754884 non-null int64
Coupon_id        1053282 non-null float64
Discount_rate    1053282 non-null object
Distance         1648881 non-null float64
Date_received    1053282 non-null float64
Date             776984 non-null float64
dtypes: float64(4), int64(2), object(1)
memory usage: 93.7+ MB
"""
ccf_online_train=pd.read_csv(r'C:\Users\LISONG\Desktop\数据分析档案\coupon_cancel\ccf_online_stage1_train.csv')

"""
ccf_online_train.info(verbose=1,null_counts=1)
可以看到shape是 (11429826, 7)，特征不多，但条数很多。
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 11429826 entries, 0 to 11429825
Data columns (total 7 columns):
User_id          11429826 non-null int64
Merchant_id      11429826 non-null int64
Action           11429826 non-null int64
Coupon_id        872357 non-null object
Discount_rate    872357 non-null object
Date_received    872357 non-null float64
Date             10773928 non-null float64
dtypes: float64(2), int64(3), object(2)
memory usage: 610.4+ MB
"""
通过观察.head()，有需要处理的地方。
1.缺失值处理。
    1.1有缺失值的分别是coupon_id，discount_rate,distance,date_received,date,四列。
        coupon_id	优惠券ID：null表示无优惠券消费，此时Discount_rate和Date_received字段无意义
        
        discount_rate优惠率：x in [0,1]代表折扣率；x:y表示满x减y。单位是元。null表示无优惠券消费
        
        coupon_id和discount_rate和date_received Non-null数量是一致的。也就是说，没有收到优惠券
        所以没有优惠券的标识，也就没有折扣利率。那么可以想见Action对应的标记一定是0.
        
        date消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本
        ；如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null
        ，则表示用优惠券消费日期，即正样本；。所以date这列反而是标记列。
ccf_offline_train['Label']=0

coupon_not_used=pd.isna(ccf_offline_train['Date']) & pd.notna(ccf_offline_train['Coupon_id'])
coupon_used=pd.notna(ccf_offline_train['Date']) & pd.notna(ccf_offline_train['Coupon_id'])
ccf_offline_train[coupon_not_used]['Label']=-1
ccf_offline_train[coupon_used]['Label']=1

def alter_label(x):
    if pd.isna(x['Coupon_id']):
        x['Label']=0
    elif pd.isna(x['Date']):
        x['Label']=1
    else:
        x['Label']=-1
lambda x 语法，x是参数，也是后面函数体中去计算和判断的根据，返回的结果根据函数体。但是这是简单函数，
所以条件设置不了太多。主要是方便使用。重要的是知道怎么用。apply()相当于把apply之前的对象按照轴形成一个Series
传给提供的函数，函数对这个轴上的每个元素执行一次操作，又将返回值放回到数据结构中。真的傻了吧唧的，早就会用了
结果折腾了一下午。
        
        distance看起来是比较独立的，但是也是最好处理的，应该说是否使用在线下和距离有很大关系。一般来说距离越远使用概率越低。
        distance是门店距离约简。即领取优惠券到最近门店的距离，分为10个等级，null为无此信息。0表示小于
        500米，10表示大于5公里，那么没有信息就不可能是0，也不可能是中位数，所以参照别人使用-1。
        ccf_offline_train[pd.isna(ccf_offline_train['Distance'])]=-1
2.discount_rate 需要处理,我认为满300减60不等于优惠20%，也不等于满100减20，这个和买的是什么商品有关。
3.Date_received Date_received领取优惠券日期 领取间隔必然会
4.Coupon_id是object ，有必要转为int
5.User_id不重复的在online中仅有76万个，相比总数1143万占比很小，大部分人是重复使用的。
6.Merchant_id和user_id的作用可以是类似于groupby，但uni

7.输出是15天内用券概率。所以既要知道他有没有用券，又要知道她什么时候用券。有没有用券分为有没有券。
但输出标记需要这么复杂吗，