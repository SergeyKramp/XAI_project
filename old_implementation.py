import operator
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import skfuzzy as fuzzy
import numpy as np
from matplotlib import pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data)
X=pd.DataFrame(iris.data,columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
#y=pd.DataFrame(iris.target,columns=['Classes'])


def compute_optimal_clusters(X , k, m=0, hard_partition=True):
  '''
  X - pandas DataFrame or numpy 2darray with the shape (N,F), where N-number of data points and F-number of features
  k - number of clusters
  m - fuzziness coeficient (defualt is for hard partitions, i.e. 0)
  '''
  if not hard_partition:
    cmeans = fuzzy.cmeans(data=X.transpose(), c=k, m=m, error=0.00001, maxiter=1000)
    prototypes = cmeans[0]
    partition_matrix = cmeans[1].transpose()
    
    # for soft clustering returns the coordinates of the prototypes and the partition matrix
    return prototypes, partition_matrix

  else:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_transform(X)
    prototypes = kmeans.cluster_centers_
    partition = kmeans.labels_

    # for hard clustering returns the coordinates of the prototypes and partitions  
    return prototypes, partition


def compute_feature_importance(X, prototypes, partitions, hard_partition=True):
    '''
    X - pandas DataFrame or numpy 2darray with the shape (N,F), where N-number of data points and F-number of features
    prototypes - a 2darray/ 2dlist with rows as the cooridinates of the prototypes
    partitions - either a list of labels of length N (hard partitions) or a partition matrix of the shape (N,Q) where Q-number of clusters (soft partitions)
    hard_partition - True for hard partitions, False for soft partitions
    '''
    #TODO: reimplement the loops and indexes in pure numpy
    X = np.array(X)
    feature_importance = []
    sum_feature_importance = 0

    if hard_partition:
      for i in range(X.shape[1]):
        ith_feature_importance = 0
        for q in range(prototypes.shape[0]):
          cluster_index = partitions == q
          Xq = X[cluster_index]
          pq = prototypes[q]
          for point in Xq:
            nominator = np.absolute(point[i] - pq[i])
            denominator = np.linalg.norm(point - pq, ord=2)
            ith_feature_importance += nominator / denominator

        feature_importance.append(ith_feature_importance)
        sum_feature_importance += ith_feature_importance       
    else: 
      for i in range(X.shape[1]):
        ith_feature_importance = 0
        for q in range(prototypes.shape[0]):
          pq = prototypes[q]
          for index, point in enumerate(X):
            meu = partitions[index, q]
            # Doesn't make sense to use norm in this case, as this is a scaler value
            #nominator = np.linalg.norm(point[i] - pq[i], ord=2)
            nominator = np.absolute(point[i] - pq[i])
            denominator = np.linalg.norm(point - pq, ord=2)
            ith_feature_importance += meu*(nominator / denominator)

        feature_importance.append(ith_feature_importance)
        sum_feature_importance += ith_feature_importance

    for i in range(len(feature_importance)):
      feature_importance[i] = (i,  feature_importance[i] / sum_feature_importance)
        

    # Returns a list of tuples with feature indices and their relative importance sorted from most important to least
    return sorted(feature_importance, key=operator.itemgetter(1), reverse=True)


def compute_performace_measures(X, k, importance, m=0, hard_partition=True):
  '''
  X - pandas DataFrame or numpy 2darray with the shape (N,F), where N-number of data points and F-number of features
  k - number of optimal clusters
  m - optimal fuzziness coefficient
  importance - list of tuples containing the features and their importance values ordered in a decreasing ranking
  hard_partition - True for hard partitions, False for soft partitions
  '''
  performance_measures = []
  X = np.array(X)
  if hard_partition:
   for i in range(len(importance) - 2):
     indices = [x[0] for x in importance[i:]]
     Xi = X[:,indices]
     kmeans = KMeans(n_clusters=k).fit(Xi)
     labels = kmeans.labels_
     score = silhouette_score(Xi, labels = labels)
     performance_measures.append(score)  

  else:
    for i in range(len(importance) - 2):
     indices = [x[0] for x in importance[i:]]
     Xi = X[:,indices]
     cmeans = fuzzy.cmeans(data=Xi.transpose(), c=k, m=m, error=0.00001, maxiter=1000)
     score = cmeans[-1]
     performance_measures.append(score)  
  
  # returns a list of performance measures
  return performance_measures

def plot_performance_graph(performance_measures):
  index = np.linspace(0, len(performance_measures) - 1, num=len(performance_measures))
  plt.plot(index, performance_measures)
  plt.scatter(index, performance_measures)
  plt.show()

prototypes, partitions = compute_optimal_clusters(X, 3, m=2, hard_partition=True)

importance = compute_feature_importance(X, prototypes, partitions, hard_partition=True)

performance_measures = compute_performace_measures(X=X, k=3, m=2, importance=importance, hard_partition=True)

plot_performance_graph(performance_measures)