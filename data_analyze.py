# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:48:41 2019

@author: lisong
"""
import pandas as pd
import numpy as np
import random as rnd
import re

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold


train = pd.read_csv(r'C:\Users\LISONG\python_script\train.csv')
test = pd.read_csv(r'C:\Users\LISONG\python_script\test.csv')
full_data = [train,test]

#开始数据清洗和规整
PassengerId = test['PassengerId']
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
#规整age，将序列中的样本的均值一倍标准差的范围内
#随机生成年龄数据填充到nan中，并且将其设为int类型，方便后面qcut
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    dataset['Age'].fillna(value=np.random.randint(age_avg - age_std, age_avg + age_std),inplace = True)
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
#利用正则表达式搜索title，方便后续使用apply
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        a=self.clf.fit(x,y).feature_importances_
        print(a)
        return a
# Class to extend XGboost classifer
def get_oof(clf, x_train, y_train, x_test):
    
    
    oof_test_skf = pd.DataFrame()

    i=0
    for train_index, test_index in kf.split(x_train):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train = pd.DataFrame(clf.predict(x_te),index=test_index)
        oof_test_skf[i] = clf.predict(x_test)
        i+=1
    oof_test = oof_test_skf.mean(axis=1)
    
    return oof_train, oof_test
# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

feature_dataframe = pd.DataFrame( {
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    },index=train.columns.values)
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1)
#这个地方feature_importance竟然无法保存，是因为helper类的feature方法仅仅是打印 而不是返回数组所以这个代码不对。目前已修改好
n_train = pd.concat([ et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train], axis=1)
n_test = pd.concat([et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test], axis=1)
#这边把get_oof的输出结果拼凑起来。用于集成学习。根据stacking的算法，这里应该是用学习器对原样例的输出作为
#次级学习器的输入，所以根本不需要保存原CV后获得的验证集的验证输出。还是说我这里的CV做错了？

以上就是全部数据整理的代码，基本上将所有属性全都变成了序值，甚至连续属性，比如Age，Fare，
拿到一个数据集，需要先确定一些东西，也就是怎么处理这些数据的问题
最主要的问题，什么模型或者算法对数据的解释性更强。
先不考虑 属性是否代表了该样本的所有样本空间，也就是说当前看到的就是需要利用到的信息
1.认识属性
    1.1阅读数据说明，从感性上认知属性的含义
    1.2查看数据取值类型，分别从定性和定量，离散和连续两个角度对特征分类，另外要注意混合型的。
    1.3数据的错误和错别字。这个很难发现，毕竟数据量大的话，拼写难以检查。
    1.4数据的缺失和null值的处理。通过describe，可以查看到null的数量，也可以看到缺失的数量。
    1.5查看承载属性值的类型，也就是方便对其进行计算机运算。但是str也显示未object,当然可以通过编写一个循环，导出所有的列的属性,并保存在一个容器里用于后续的使用和观察。
    1.6获得数据的分布distribution，通过describe来查看，describe return 一个df，因此可用于保存。
        查看分布也需要用到一些领域知识，domain。
    1.7数据distribution。describe 是一个强大的功能，可以指定包含和排除的类型，并且会自动区分定性和定量的属性，生成不同的观察值。
    
2.基于数据分析的假设（对属性的处理）
    也就是基于上面数据属性值类型的分类，数据分布，领域知识，数据说明等等对数据做出的调整。
    2.1相关性，correlating即查看属性和标记的相关性高低，我们肯定希望选择相关性更强的属性。
    2.2完整性，completing,对缺失值的处理，缺失值将影响模型的质量。
    2.3校正,correcting，逻辑上相关性低的，缺失值太多的，非标准化的数据的处理等。
    2.4创造新属性creating，可以是校正，连续属性离散化或离散属性连续化，或者整合几个属性，总之是提升可计算的程度。
    2.5分类classifying？？根据领域知识建立假设，找到和目标最相关的属性或属性集，这似乎是相关性的内容。
    强行变成5C？

3.基于上述原则的分析。
    3.1categorical属性的分析，基本上根据标记，使用groupby.mean()就可看出属性值和标记的强弱关系。记录下这些关系
        还可以使用可视化方法对这些假设做验证。应该就是为了写报告吧。
        3.1.1直方图可以直观的对连续属性划分面元后观察数据的分布。在划分面元时，可以对一些特殊区间保留偏好。
            比如婴儿，老人的定义。这个完全可以由使用者来定义。记录下这个直方图的特点。
            **seaborns的学习

1.这些属性都是什么类型，都是float，还是有特殊类型，当然大部分不是数字，就是str，
或者说确定这一属性的取值集合，是连续还是离散，连续是序列的连续还是实数上的连续，离散是
有必要将其连续化吗，怎样的连续化更好。
2.属性的取值是否有缺失，0值怎么处理，
3.了解特征的首要途径是看数据描述，其中会对数据的含义做出说明。其次看属性的取值，分别为nominal标称的，名词性的，即文本型的，
ordinal 序数，代表序列的，比如男女，正反，标称和序数统称为定性的qualitative，或分类的categorical,interval,区间的，即属于某一区间，
ratio，比率的，也是属于某一区间，这些称为定量的quantitative或数值的numerical
对定性和定量做区分，也可以方便后面对数据的视觉化处理。序列是指其数值从高到低是有意义的。
属性也可以按照连续和离散来区分，这个相关概念也很好理解，分为也是为了后续的视觉化处理。
还有混合的比如这里的Ticket，其中有数字和字母混合在一起。还有一种字母数字，alphanumerical，像Carbin舱室的名称。

