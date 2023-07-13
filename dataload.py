import numpy as np
from measure import *
import arff
from numpy.linalg import matrix_rank


# yeast
'''
data = scio.loadmat('./data/yeast.mat')
# print(data)
X_train = np.array(data['Xapp'])
Y_train = np.array(data['Yapp'])
X_test = np.array(data['Xgen'])
Y_test = np.array(data['Ygen'])

X = np.r_[X_train, X_test]
Y = np.r_[Y_train, Y_test]
num_train = X_train.shape[0]
num_feature = X_train.shape[1]
num_label = Y_train.shape[1]
#scio.savemat('./data/matlabdata/yeast.mat',{'X':X, 'Y': Y} )

print('yeast')

#'''

# image
'''
data = scio.loadmat('./data/image.mat')
X = np.array(data['data'])
Y = np.array(data['target']).T
X_train = X[:][:1800]
Y_train = Y[:][:1800]
X_test = X[:][1800:]
Y_test = Y[:][1800:]
num_train = X_train.shape[0]
num_feature = X_train.shape[1]
num_label = Y_train.shape[1]
print('images')
#'''

# birds
'''
file = (open('./data/birds.arff', 'r'))
decoder = arff.ArffDecoder()
draft = decoder.decode(file)
file.close()
data = np.array(draft['data'], dtype=np.float32)
X = data[:, 0: 260]
Y = data[:, 260:]
X_train = X[:595]
Y_train = Y[:595]
X_test = X[595:]
Y_test = Y[595:]
num_train = X_train.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('birds')

#scio.savemat('./data/matlabdata/birds.mat',{'X':X, 'Y': Y} )

#'''

# CAL500
#'''
draft = arff.ArffDecoder().decode(open('./data/CAL500.arff', 'r'))
data = np.array(draft['data'], dtype=np.float32)
X = data[:, : 68]
Y = data[:, 68:]
X_train = X[:462]
Y_train = Y[:462]
X_test = X[432:]
Y_test = Y[432:]
num_train = X_train.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('CAL500')

#scio.savemat('./data/matlabdata/CAL500.mat', {'X':X, 'Y': Y} )

#'''


# emotions
'''
draft1 = arff.ArffDecoder().decode(open('./data/emotions.arff', 'r'))
data1 = np.array(draft1['data'], dtype=np.float32)
X = data1[:, :72]
Y = data1[:, 72:]
X_train = data1[:510, :72]
Y_train = data1[:510, 72:]
draft2 = arff.ArffDecoder().decode(open('./data/emotions-test.arff', 'r'))
data2 = np.array(draft2['data'], dtype=np.float32)
X_test = data1[500:, :72]
Y_test = data1[500:, 72:]
num_train = X_train.shape[0]
num_feature = X_train.shape[1]
num_label = Y_train.shape[1]
print('emotions')
#scio.savemat('./data/matlabdata/emotions.mat',{'X':X, 'Y': Y} )

#'''


# mediamill
'''
draft1 = arff.ArffDecoder().decode(open('./data/mediamill-train.arff', 'r'))
data1 = np.array(draft1['data'], dtype=np.float32)
X_train = data1[:, :120]
Y_train = data1[:, 120:]
draft2 = arff.ArffDecoder().decode(open('./data/mediamill-test.arff', 'r'))
data2 = np.array(draft2['data'], dtype=np.float32)
X_test = data2[:, :120]
Y_test = data2[:, 120:]
X = np.r_[X_train, X_test]
Y = np.r_[Y_train, Y_test]
num_train = X_train.shape[0]
num_feature = X_train.shape[1]
num_label = Y_train.shape[1]
print('mediamill')
#scio.savemat('./data/matlabdata/mediamill.mat',{'X':X, 'Y': Y} )

#'''


# medical
'''
draft1 = arff.ArffDecoder().decode(open('./data/medical.arff', 'r'))
data1 = np.array(draft1['data'], dtype=np.float32)
X = data1[:, :1449]
Y = data1[:, 1449:]

num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('medical')
#scio.savemat('./data/matlabdata/medical.mat',{'X':X, 'Y': Y} )

#'''

# scene
'''
draft1 = arff.ArffDecoder().decode(open('./data/scene-train.arff', 'r'))
data1 = np.array(draft1['data'], dtype=np.float32)
X_train = data1[:, :294]
Y_train = data1[:, 294:]
draft2 = arff.ArffDecoder().decode(open('./data/scene-test.arff', 'r'))
data2 = np.array(draft2['data'], dtype=np.float32)
X_test = data2[:, :294]
Y_test = data2[:, 294:]

X = np.r_[X_train, X_test]
Y = np.r_[Y_train, Y_test]
num_train = X_train.shape[0]
num_feature = X_train.shape[1]
num_label = Y_train.shape[1]
print('scene')
#scio.savemat('./data/matlabdata/scene.mat',{'X':X, 'Y': Y} )

#print(X_test.shape[0])
#'''


# delicious * 没法
'''
draft1 = arff.ArffDecoder().decode(open('./data/delicious.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :500]
Y = data1[:, 500:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('delicious')
#scio.savemat('./data/matlabdata/delicious.mat',{'X':X, 'Y': Y} )

#'''


# enron *
'''
draft1 = arff.ArffDecoder().decode(open('./data/enron-train.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X_train = data1[:, :1001]
Y_train = data1[:, 1001:]
draft2 = arff.ArffDecoder().decode(open('./data/enron-test.arff', 'r'))
data2 = np.array(draft1['data'], dtype=int)
X_test = data2[:, :1001]
Y_test = data2[:, 1001:]

X = np.r_[X_train, X_test]
Y = np.r_[Y_train, Y_test]
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('enron')
scio.savemat('./data/matlabdata/enron.mat',{'X':X, 'Y': Y} )

#'''

# Corel5k *
'''
draft1 = arff.ArffDecoder().decode(open('./data/Corel5k.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :499]
Y = data1[:, 499:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('corel5k')
#scio.savemat('./data/matlabdata/Corel5k.mat',{'X':X, 'Y': Y} )

#'''

# tmc2007 *
'''
draft1 = arff.ArffDecoder().decode(open('./data/tmc2007.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :499]
Y = data1[:, 499:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('tmc2007')
scio.savemat('./data/matlabdata/tmc2007.mat',{'X':X, 'Y': Y} )

#'''

# bibtex *
'''
draft1 = arff.ArffDecoder().decode(open('./data/bibtex-train.arff', 'r'))
data1 = np.array(draft1['data'],dtype=int)
X_train = data1[:, :1836]
Y_train = data1[:, 1836:]
draft2 = arff.ArffDecoder().decode(open('./data/bibtex-test.arff', 'r'))
data2 = np.array(draft1['data'],dtype=int)
X_test = data2[:, :1836]
Y_test = data2[:, 1836:]

X = np.r_[X_train, X_test]
Y = np.r_[Y_train, Y_test]
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('bibtex')
#scio.savemat('./data/matlabdata/bibtex.mat',{'X':X, 'Y': Y} )

#'''


# slashdot *
'''
draft1 = arff.ArffDecoder().decode(open('./data/SLASHDOT-F.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, 22:]
Y = data1[:, :22]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('slashdot')
scio.savemat('./data/matlabdata/slashdot.mat',{'X':X, 'Y': Y} )

#'''


# language log *
'''
draft1 = arff.ArffDecoder().decode(open('./data/LLOG-F.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, 75:]
Y = data1[:, :75]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('language log')
#scio.savemat('./data/matlabdata/language_log.mat',{'X':X, 'Y': Y} )

#'''

# Corel16k001 *
'''
draft1 = arff.ArffDecoder().decode(open('./data/Corel16k001.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :500]
Y = data1[:, 500:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('corel16k001')
#scio.savemat('./data/matlabdata/corel16k001.mat',{'X':X, 'Y': Y} )

#'''

# flag
'''
draft1 = arff.ArffDecoder().decode(open('./data/flags.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :22]
Y = data1[:, 22:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('flag')
#scio.savemat('./data/matlabdata/language_log.mat',{'X':X, 'Y': Y} )
#'''

'''
X = np.array([[1,1,1],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,1,1],[1,1,1],[1,1,1],
                 [1,1,1],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,1,1],[1,1,1],[1,1,1],
                 [1,1,1],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,1,1],[1,1,1],[1,1,1],
                 [1,1,1],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,1,1],[1,1,1],[1,1,1]])
Y = X.copy()
'''

# eurlex-eurovoc
'''
draft1 = arff.ArffDecoder().decode(open('./data/eurlex-eurovoc-descriptors/eurlex-ev-fold1-train.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :-3993]
Y = data1[:, -3993:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('eurlex-eurovoc')
#'''

# eurlex-eurovoc
'''
draft1 = arff.ArffDecoder().decode(open('./data/eurlex-eurovoc-descriptors/eurlex-ev-fold1-train.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :-3993]
Y = data1[:, -3993:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('eurlex-eurovoc')
#'''

# eurlex-subject
'''
draft1 = arff.ArffDecoder().decode(open('./data/eurlex-subject-matters/eurlex-sm-fold1-train.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :-201]
Y = data1[:, -201:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('eurlex-subject')
#'''


# eurlex-directory
'''
draft1 = arff.ArffDecoder().decode(open('./data/eurlex-subject-matters/eurlex-sm-fold1-train.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :-412]
Y = data1[:, -412:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('eurlex-directory')
#'''

# bookmask
'''
draft1 = arff.ArffDecoder().decode(open('./data/bookmarks.arff', 'r'))
data1 = np.array(draft1['data'], dtype=int)
X = data1[:, :-208]
Y = data1[:, -208:]
#print(data1)
num_train = X.shape[0]
num_feature = X.shape[1]
num_label = Y.shape[1]
print('bookmask')
#'''


#print(Y.shape[1])
#print(matrix_rank(Y))
#print(X.shape[0])
#np.random.seed(1)