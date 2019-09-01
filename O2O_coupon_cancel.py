
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
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from datetime import datetime
#通过os.chdir(r'C:\Users\Frank_li\python_script')来改变os.getcwd()获取的默认位置。
ccf_offline_test=pd.read_csv(os.path.join(os.getcwd(),'python_script\O2Ocoupon_cancel\ccf_offline_stage1_test_revised.csv'))
coupon_sample = pd.read_csv(os.path.join(os.getcwd(),'python_script\O2Ocoupon_cancel\sample_submission.csv'),header=0)
ccf_offline_train=pd.read_csv(os.path.join(os.getcwd(),'python_script\O2Ocoupon_cancel\ccf_offline_stage1_train.csv'))
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
ccf_online_train=pd.read_csv(os.path.join(os.getcwd(),'python_script\O2Ocoupon_cancel\ccf_online_stage1_train.csv'))
dataset=[ccf_offline_train,ccf_online_train,ccf_offline_test]
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
1.特征工程。
    主要是对数据表中的特征项进行调整。根据.info()查看，有缺失值的分别是
    coupon_id，discount_rate,distance,date_received,date,四列。但根
    据文档描述，缺失值并不是无意义的。因此需要结合文档对特征进行处理。另外，
    特征中，并没有标记，所以需要自行设定。
    
1.1缺失值处理。
        有缺失值的分别是coupon_id，discount_rate,distance,date_received,date,四列。
        1.coupon_id	优惠券ID：null表示无优惠券消费，此时Discount_rate和Date_received字段无意义。
         coupon_id和discount_rate和date_received Non-null数量是一致的。也就是说，没有收参照别人使用到优惠券
        所以没有优惠券的标识，也就没有折扣利率。那么可以想见Action对应的标记一定是0.
        Coupon_id是object ，有必要转为int
def Cid(x):
    if pd.isna(x['Coupon_id']):
        return 0
    elif x['Coupon_id']=='fixed':#fixed表示限时低价活动。非固定优惠。
        return 1
    else:
        return 2

将na去掉，以上调整函数的意思是无消费券可用，标记是-1，fixed表示是特殊记号用0表示，其他的则是1表示。
这里需要使用新列，不能替换原列。offline是float，Online是object，需要调整。

        2.discount_rate优惠率：x in [0,1]代表折扣率；x:y表示满x减y。单位是元。null表示无优惠券消费
        需要处理,我认为满300减60不等于优惠20%，也不等于满100减20，这个和买的是什么商品有关。归一化容易
        统一量纲，但可能也会损失一些信息。但如何评估这个损失呢。暂时不好讲。另外有多达40%的数据没有使用
        优惠券，因此这部分应该设为-1.
def dr(x):
    if pd.isna(x['Discount_rate']):
        return 0
    elif ':' in x['Discount_rate']:
        return float(x['Discount_rate'].split(':')[1])/float(x['Discount_rate'].split(':')[0])
    elif x['Discount_rate']=='fixed':
        return 0.8
    else:
        return float(x['Discount_rate'])

        3.distance看起来是比较独立的，但是也是最好处理的，应该说是否使用在线下和距离有很大关系。一般来说距离越远使用概率越低。
        distance是门店距离约简。即领取优惠券到最近门店的距离，分为10个等级，null为无此信息。0表示小于
        500米，10表示大于5公里，那么没有信息就不可能是0，也不可能是中位数，所以设为-1。


       4.Date_received Date_received领取优惠券日期 这个与Date构成了Interval，而且补全他的NA
       值意义不大，一是其与总数据少了2个数量级，二是没有好的填充方式。
           因为test数据集里有date_received特征，因此应该想办法利用起来，Label作为辅助可用。
def to_weekday(x):
    return pd.to_datetime(str(int(x))).weekday()+1
   
def d_received(x):
    if pd.isna(x['Date_received']):
        return 0
    else:
        return to_weekday(x['Date_received'])
       
       5.
       
        
1.2新特征
        date消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本
        ；如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null
        ，则表示用优惠券消费日期，即正样本；。data里没有Date=null & coupon_id =null的情形。
        所以date这列反而是标记列。
设定标签。作为标记向量。因为没有没优惠群且没消费的记录。所以这里收录的样本要么有优惠券，要么有消费。实际上
我们的任务就是通过这一系列的特征，找到预计会在哪一个时间段消费。
offline里只有40%的样本发生了交易。online则有近9成的样本发生了交易。
def Label(x):
    if pd.isna(x['Coupon_id']):
        return 0
    elif pd.notna(x['Date']):
        return 1
    else:
        return -1
1.3专属特征的整合
        Distance和Aciton分别是2个数据集独有的，那么如何利用这些信息。关键就是理解这个特征的含义。
        而且因为客观原因，其与Label的关系需要用图来确认。
       1. Action.属于online特有的属性。0 点击， 1购买，2领取优惠券，而且记录数完整。使用起来和Distance是
       类似的。Distance和ACtion很显然与最终的消费或者说我们定义的Label是相关的。相乘是不科学的。使用优惠券
       购买东西的最大的可能性一定是自己购买了优惠券。
       2.distance看起来是比较独立的，但是也是最好处理的，应该说是否使用在线下和距离有很大关系。一般来说距离越远使用概率越低。
        distance是门店距离约简。即领取优惠券到最近门店的距离，分为10个等级，null为无此信息。0表示小于
        500米，10表示大于5公里，那么没有信息就不可能是0，也不可能是中位数，所以设为-1。
        这两个属性可以作为独立属性来看待，因为线上和线下有可能同时进行，但也很可能是分开进行。
        那么2个数据集应该怎么处理呢，共同点是User_id和Merchant_id有重合，区别是大小不一样。消费时间有重合。
        那么合并这2个数据集的话。那就根据自己的假设来调整数据集。
        

def Action(x):
    act_dict={1:[0,1,2,10],2:[3,4,5,6,7,8,9]}
    if pd.isna(x['Distance']):
        return 0
    elif x['Distance'] in act_dict[1]:
        return 1
    else: 
        return 2
    

g=sns.FacetGrid(ccf_offline_train,col='Label')
g.map(plt.hist,'distance')

ga=sns.FacetGrid(ccf_offline_train,col='Label')
ga.map(plt.hist,'Action')
若按照我们的标记来划分，则看不出趋势。但总体与我的猜测一致，类似于微笑曲线，因为线下的人
很可能先获得了优惠券，再去线下消费，而不是仅仅站在门口才消费。
根据我们的调整，看起来Action=1占Label=1的比例变的大了，看起来更加标准化。因此决定这样去整合两个数据集。


1.4标记处理
创造interval这一特征，表示领取到消费的时间，算法是Date-date_received,但是无论二者谁是Nan，得到的结果都是Nan
虽然希望来区分，但为了简化输出标记，则所有nan都视为0，大于我们考察的周期的设为2，小于我们考察的周期的设为1.
Label并不是标记，而只是为了利用Date，date_received,coupon_id而创造出来的新特征。真实标记应该是output_label，
因为学习任务是输出15天内用券概率。这样无论是logical regressor 还是bayes都可以了 但不管怎样都不能有nan值。

ccf_offline_train['interval']=ccf_offline_train['Date']-ccf_offline_train['Date_received']
ccf_online_train['interval']=ccf_online_train['Date']-ccf_online_train['Date_received']

def output_label(x,consum_period=15):
    #可以修改输出标记，根据你需要的日期来修正，这一列是作为y的。
    if pd.isna(x['interval']):
        return 0
    elif x['interval']>consum_period:
        return 2
    else:
        return 1
    
1.5无缺失不连续特征
1.User_id不重复的在online中仅有76万个，相比总数1143万占比很小，大部分人是重复使用的。
2.Merchant_id和user_id的作用可以是类似于index，这类无特别含义的信息，不放入模型中。groupby这个范围还是太大了。

1.6完成数据整理并保存
func_list=[Cid,dr,Label,output_label,d_received,Action]        
#off 需要cid,dr,label,output_label,d_received,action
#online 需要cid,dr,label,output_label,d_received
#test 因为是线下 需要cid,dr,action,d_received
#很忧伤，竟然用不了label，不过用上了d_received
def tidy_data(df,f):
    return df.apply(f,axis=1)
#为了考虑性能打印一个程序运行时间。因为传入该函数的参数是列表，是可变对象，因此会直接修改该列表
#所以考虑性能占用，不需要去修改这里的运行方式。
def prepare(dataset,func_list):
    begin=time.clock()
    for i in range(len(func_list)):
        dataset[0][func_list[i].__name__]=tidy_data(dataset[0],func_list[i])
    for i in range(len(func_list)-1):
        dataset[1][func_list[i].__name__]=tidy_data(dataset[1],func_list[i])
    for i in [0,1,4,5]:
        dataset[2][func_list[i].__name__]=tidy_data(dataset[2],func_list[i])
    end=time.clock()
    print('Total running time:{} s'.format(end-begin))
    return dataset
dataset=prepare(dataset,func_list)
#丢弃不需要的特征
dataset[0].drop(columns=['Coupon_id','Discount_rate','Distance','Date','interval','Date_received','Label'],inplace=True)
dataset[1].drop(columns=['Coupon_id','Discount_rate','Date','interval','Date_received','Label'],inplace=True)
dataset[2].drop(columns=['Coupon_id', 'Discount_rate', 'Distance','Date_received'],inplace=True)
#保存新的数据集到本地
dataset[0].to_csv(os.path.join(os.getcwd(),'python_script\O2Ocoupon_cancel\offline_train.csv'))
dataset[1].to_csv(os.path.join(os.getcwd(),'python_script\O2Ocoupon_cancel\online_train.csv'))
dataset[2].to_csv(os.path.join(os.getcwd(),'python_script\O2Ocoupon_cancel\\off_test.csv'))



以上完成特征处理。

2.模型选择
    利用sklearn中多个已搭建好的模型和cross-validation来对数据集进行分别的训练，最后选择验证表现最好的模型。

#有了X和y和pred就能适用模型了，在DecisionTree上简单验证得到了0.98的分数。说明还凑合哦。
X=pd.concat([dataset[0],dataset[1]],axis=0,sort=False)
y=X['output_label']
X.drop(columns=['Merchant_id','User_id','output_label'],inplace=True)
pred=dataset[2].drop(columns=['User_id','Merchant_id'])
2.1构建多模型训练结果

2.2画出每一个模型的auc曲线，获得auc值。

2.3决定选择的结果。

2.4提交结果。
    观察test的已有特征，没有date，或者说date就是我们需要输出的内容。那么只有Date_received的话，或者将其转为星期
    数或者节假日标记，再计算会更好。但为了简便起见，暂时忽略这块。
    其次，在没有外部信息的话，只有决策树，贝叶斯能利用Merchant_id，user_id，date_received，其他模型则不需要利用
    这些，这些只是作为提交的结果的组成部分。
    
    在模型上的验证。输出曲线。
lambda x 语法，x是参数，也是后面函数体中去计算和判断的根据，返回的结果根据函数体。但是这是简单函数，
所以条件设置不了太多。主要是方便使用。重要的是知道怎么用。apply()相当于把apply之前的对象按照轴形成一个Series
传给提供的函数，函数对这个轴上的每个元素执行一次操作，又将返回值放回到数据结构中。真的傻了吧唧的，早就会用了
结果折腾了一下午。

函数参数设置中，*号是作为解压操作符，在参数时，否则是乘法操作符。带*号,在函数定义的时候的含义是，不限数量的参数。
在调用函数时，在参数前加*，表示为传入的这个可迭代对象解压，迭代对象中的元素作为参数传入函数体。**kwags，代表的是
可以传入不限长的字典，这个字典在调用时也需要使用**操作符，告诉python需要把字典中的每个键值对作为参数传入函数体。
在函数定义时的**只表示，可以接受不定长的字典。那么不传入字典也没关系。所以并不是默认的无意义的。还是和用法有关。