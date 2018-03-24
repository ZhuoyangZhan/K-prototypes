import pandas as pd
import numpy as np
from kmodes import kmodes
from kmodes import kprototypes
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

"""
由于本数据集中既有连续型变量，又有类别变量，因此考虑使用K-prototype算法来进行聚类。K-prototype的本质就是用K-means处理连续型变量，
用K-modes处理类别变量，然后两类结果合起来即为prototype.根据Zhexue Huang在其论文《Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values》中提到，
在计算混合类型数据簇中心点时，连续型变量的均值即为连续型数据原型的中心，类别变量中取值频率最高的属性即为类别型数据原型的中心。两者相结合即为簇的中心
在计算相似度量时，K-prototype使用相异度（Dissimilarity）来衡量数据点之间的相似程度。
在K-prototype算法中混合属性的相异度分为属性属性和分类属性，两者分开求然后相加。
对于数值属性:使用欧几里得距离
D(numerical) = |x(i) - centroid(i)|2 
对于分类属性:使用海明威距离，即属性值相同，为0 ；属性值不同，为1。
D(Categorical) = 1 if x != centroid   or  
               = 0 if x == centroid
因此数据点和某一簇之间的相异度为：
D(x(i), centroid(i)) = sum(D(numerical)) + u*sum(D(Categorical)), 其中u为分类属性的权重因子
要minimize的目标函数为
C = sum(sum(u(il)*D(x(i),centroid(l))), 其中u = 1 when x(i)在centroid（l）里面， 反之则u = 0

以上为K-prototype计算距离，中心点以及优化目标函数的基本原理，在本次Assignment中，我直接调用Python中kmode包里的Kprototypes函数，其计算
原理和以上基本相同。在具体作业中，首先我将连续型变量和类别变量分开，然后对连续型变量进行标准化统一化，然后再将两类数据合起来放入Kprototype函数中
进行计算。函数的输入为样本数据和聚类数，输出为样本类别标签、各个类别样本数以及聚类中心。
具体步骤是：
输入：聚类簇的个数k， 权重因子
输出：产生好的聚类。
步骤：从数据集中随机选取k个对象作为初始的k个簇的原型
遍历数据集中的每一个数据，计算数据与k个簇的相异度。然后将该数据分配到相异度最小的对应的簇中，每次分配完毕后，更新簇的原型
然后计算目标函数，然后对比目标函数值是否改变，循环直到目标函数值不在变化为止。

目前函数最主要的不足之处是在处理大量数据时运算速度较慢。其原因是K-prototype的时间复杂度为 O(i*k*n*d),
i = 迭代次数, k = 中心数, n = 数据量， d = 特征维数. 目前我想到可以改进的地方是对连续变量进行PCA降维，减少d特征维数。

"""

def kprototype(filename, num_clusters):
    #输入数据
    #若输入完整数据库30000个entries可能计算时间会过久，故先以3000个data points作为例子。
    num_data = 3000
    X_original = np.genfromtxt(filename, dtype=object, delimiter=',')[1:num_data,:]
    
    #normalize连续型变量
    X_categorical = X_original[:,0:10]
    X_numerical = normalize(X_original[:,11:], norm='l2')
    #对于连续型变量，如果数量较多的话可以考虑使用PCA降维
    #X_numerical = PCA(n_components=1).fit_transform(X_numerical)
    X = np.concatenate((X_categorical, X_numerical), axis=1)

    #开始训练，默认权重u为0.5 * 连续型变量值的标准差
    kproto = kprototypes.KPrototypes(n_clusters=num_clusters, init='Cao', verbose = 2)
    clusters = kproto.fit_predict(X, categorical=[0,1,2,3,4,5,6,7,8,9])
    print '\n'
    
    #输出每个数据点所属的聚类标签
    print 'Labels of each data point: \n', kproto.labels_, '\n'
    
    #输出各个类别的样本数
    for i in range(num_clusters):
        num_sample = 0
        for n in kproto.labels_:
            if i == n:
                num_sample += 1
        print 'numbers of samples in the', i, 'cluster: ', num_sample
    print '\n'
    
    #输出聚类中心
    print 'Clusters: \n', kproto.cluster_centroids_
    
    #输出目标函数成本
    print 'Cost: ', kproto.cost_

#输入数据集和类别数，开始计算K-prototypes    
filename = 'samples.csv'
num_cluster = 3
kprototype(filename, num_cluster)

