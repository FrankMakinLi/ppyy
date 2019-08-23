# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:49:49 2019

@author: lisong
"""

seaborn学习,貌似我搞错了，先从复杂图形学起了。FacetGrid一旦调用.map()方法就会报错，但直接用
relplot这个最普通的接口，却可以实现FacetGrid的作图功能，而且实际上relplot的类型仍然是Facetgrid
所以问题就在于.map这个函数。如果避免使用这个方法，就可以避免报错。
1.sns.FacetGrid
    如果需要基于同一变量的不同取值条件下，所要考察的另外不同变量的分布或不同变量之间的关系，则可以使用
    FacetGrid这个类，他可以很方便的调用matplotlib的画布功能，并按照你的预期将图画在画布上。首先需要创建
    一个Facet实例，由df的数据结构作为数据源。在其中定义的col和row参数，对应的是画布的坐标轴。而不是图形的坐标轴。
    所以理论上说，他可以考察最多四位数据，当然这也是因为2-D图形只能表示2维数据。hue参数不同于col和row，相当于
    在原画布上加了一层维度，换句话说，矩阵上每幅图多了一个维度。
    在作画时，调用的是map方法，map方法使用plt的图样，再传入相关参数即可。
    
2.sns.PairGrid
    类似于散点图矩阵，将数据中的属性两两配对，形成矩阵，但这个很显然只适用于numerical型特征。但不局限于散点图？
    用法类似，先初始化一个PairGrid实例，然后使用.map方法来画图。可以设置对角线和非对角线的图形类别，比如对角线
    用直方图，非对角线散点图。
    可以通过hue参数指定一个categorical的特征，在pairgrid中按照不同值上色。
    默认是所有的numerical特征都会参加pair，但也可以指定想要观察的pair对。
    还可以对矩阵的上三角、下三角、和对角线分别设置不同类型的图形。
    
3.sns.relplot relational plot
    这是一个基本的类，用来对两个变量的关系做图。可以指定hue和style参数，来增加数据的区分度，或者说引入新的维度。
    hue可以理解特征值的类型，如果是离散的，会默认给出不同对比度的颜色，如果是连续属性，则会给出连续颜色区分。
    size参数也可以被定义来区分散点，可以根据size的取值来显示不同的点的大小。
    relplot被默认设为散点图，但实际上这是一个通用入口，也可以改变图形类型，kind参数用来调整。
    通过col这个参数，可以将其转为FacetGrid。但不同于FacetGrid方便。relplot()是base于FacetGrid的。
    
4.sns.lineplot
    这是一个默认用line图的入口。lineplot会自动对x轴的数据sort，也可以禁用sort。
    除了表示趋势之外，还可以绘制趋势的置信区间，但若样本点过多，会使得运行变慢，因此也可以取消这个设置。
    除了置信区间，还可以使用标准偏差来代表，因为置信区间使用bootstrapping，所以会变慢，但标准偏差相对资源消耗少。
    还可以关掉学习器，通过estimator这个参数来控制。如果不关，图形看起来会相对平滑一些。
    线图的hue参数可以将一条线分为两条或多条。同理，styLe,size等参数也可以再区分或customize
    时间序列，如果检测到传入的是时间序列，那么虽然是线图，但实际上数据会传递到一些underlying绘图函数中，
    这点与一般线图不同。

5.sns.catplot() categorical plot
    这是一个默认用来表示离散属性的图。但与散点图有真正不同，因为在catplot里，所有属性会被归类在一个分类里，
    而不是分布在整个坐标系里。stripplot()是catplot()的default kind，在同一个位置的点会被调整为一些随机分布
    看起来不是一条直线的位置上。jitter属性控制这个渲染。if false 那么，就会变成一条直线。
    swarmplot()提供了一个算法，使得在0轴上各个分类的点不重合，这样可以更加清楚的展示数据的分布，参数是
    kind='swarm'。提供更高维度只有hue参数，style和size参数不被提供。ordinal的属性，哪怕看起来numerical
    一样也能被category。
    boxplot()箱型图，kind='box'，箱型图也有hue参数，在一个位置分出2个箱体。
    boxenplot()和一般的box类似，kind='boxen'，可以更好的展示数据的分布，也更美观。很适合large dataset
    violinplot()，kind='violin'，用密度表示箱体的图。
    barplot()，条形图/直方图？kind='bar'，
    countplot()，kind='count'，在此类似于直方图，只需在x轴或y轴上定义一个离散属性。
    pointplot(),kind='point'，要点图，会计算置信区间，把各个分类的要点连在一起。看起来会更加清晰。
    catplot()也是基于FacetGrid的，所以也方便的引入参数col,row to transforming the plot to Facet
    
6.distplot()distribution plot 分布图。
    distplot()是主要接口。默认是histogram类型的，使用KDE 即kernel density estimate,核密度估计来
    绘图。hist会自动为数据的宽度划分bins，来适配整体的宽度。
    rugplot()，rug=True，小地毯图，也就是在横轴上添加tick来显示当前取值上的密度。
    distplot()用的是hist=Fasle,kde=False这样的参数来指定图的种类。kde计算占用更多，因为每个观测值被
    标准正态分布中心化了，并放置到图上。在KDE图中，bandwidth参数控制标准正态曲线和样本分布的偏离度，偏离度越
    小，实际上曲线越不光滑，但也更好的展示样本的分布。
    
7.jointplot()二元分布图，或者称为连接图，用一种多面图来展示两个变量之间的关系，并在单独的轴上展示其独立分布。
    scatterplot是jointplot的默认图。
    hexplot()，kind='hex'和直方图histogram类似的，用一个六边形组成的bins来组成图形。
    kind='kde',与distplot一样，jointplot也有kde参数，其展示的就是数据的密度和等高线。
    查看分布还是得用int或float值,或者至少是ordinal，如果是离散值还是用catplot
    也可以先通过kdeplot()画图，再加入rugplot()到同一个图像上，那样的话，需要先初始化一个ax实例，通过plt创建。
    jointplot是基于JointGrid的类，所以也可以直接使用JointGrid来创建。
    
8.sns.pairplot()是基于PairGrid()的类
    使用方法于PairGrid类似，为什么使用.map()会报错，因为为了更好的使用内存，使用的是迭代器和yield机制，
    所以参数只保存在运行时的内存里，因此map()和创建实例必须同时进行。包括设置参数，如果是图片对象已经创建
    再设置参数，实际上不会对这个图片再有影响，因为这个实例所用到的参数都已经迭代完了。
    
9.sns.regplot(),sns.lmplot()，用于表示两个变量之间回归关系的图但是二者有区别，具体后面再说。一般情形下，
会使用scatterplot以及一个线性回归线，并且提供一个95%的置信区间wideth
    regplot接受各种各样的数据的参数类型，比如np.darray,pd.series,pd.df，而且不用指明data这个参数，但是
    lmplot是不行的，必须接受data这个参数，而且他更倾向于接收long-term参数。
    data可以接受ordinal参数，但其他类型的离散值是不行的。
    在划回归趋势线时，可以提供order参数来改变回归函数的幂，比如二次函数，更加显著的拟合。
    robust参数提供对于outliers异常值的修正，不至于使得一些异常值对趋势线影响过大。
    logistic参数用来对二元Y值的调整。但实际上这个很傻不是吗，需要使用科学计算的都会占用过大计算资源，
    可以通过ci参数来关闭。
    residplot()残差点阵图，为每个点生成一个残差值，并绘制在图上。
    如果想要知道一个函数内第三个变量变动对这两个变量的关系有何影响的，这也是lmplot和regplot不同的地方
    regplot只能提供简单的回归图形。复杂任务只能交给lmplot，比如说指定hue参数，可以生成2个趋势线。
    所以类似于FacetGrid,也可以提供col和row参数来增加数据的分类。
    regplot和lmplot默认图像大小不同在于，regplot调用的是plt.ax里的框架，而lmplot调用的是FacetGrid的框架
    二者所获得的默认参数是不同的。
    kind='reg'可以用于其他可接受类似参数的multiple-plot，这也是lmplot和regplot区别的原因吧。
    
总算看完了文档，后面就是熟练使用这个框架，毕竟相对于其他用途，他的一些功能简单实用，比matplotlib要好用多了
目前要看完sklearn学习笔记。
    