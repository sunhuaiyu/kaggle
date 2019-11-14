from pylab import *
import pandas as pd; from pandas import Series, DataFrame
from sklearn import svm
from random import shuffle
from sklearn.preprocessing import scale

def pca(x):
    u, z, v = svd(x)
    z = z[z>0.0001]
    k = len(z)
    return u[:, :k].dot(diag(z).dot(v[:k,:k]))


m = pd.read_csv('train.csv', index_col='Id')
X = scale(m.ix[:, :-1].astype(float), axis=0)
y = m.ix[:, -1]
# m1 = scale(m.ix[:, 'Elevation':'Horizontal_Distance_To_Fire_Points'].astype(float), axis=0)
# X = hstack((m1, m.ix[:, 'Wilderness_Area1':'Soil_Type40'].astype(float))) #15120x54

a = pd.read_csv('test.csv', index_col='Id')
A = scale(a.ix[:, :].astype(float), axis=0)
# a1 = scale(a.ix[:, 'Elevation':'Horizontal_Distance_To_Fire_Points'].astype(float), axis=0)
# A = hstack((a1, a.ix[:, 'Wilderness_Area1':'Soil_Type40'].astype(float)))

f = svm.SVC(C=1.8)
f.fit(X, y)
p = f.predict(A)

s = pd.read_csv('sampleSubmission.csv', index_col='Id')
s.Cover_Type = p
s.to_csv('sampleSubmission.csv')



score = []
# train

for c in arange(0.8, 2.8, 0.2):
    for g in arange(0, 1., 0.2):
        f = svm.SVC(C=c, gamma=g, class_weight='auto')
        f.fit(X[:5000], y[:5000])
        score.append([c, g, f.score(X[5000:], y[5000:])])

score = array(score).round(2)
for i in score: print i
    
DataFrame(score).to_csv('tmp.txt', index=None)


