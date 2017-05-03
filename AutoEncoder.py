#coding :utf-8

import sys
import os
import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import datetime
from compiler.ast import flatten
from collections import deque
import random
from sklearn.cross_validation import train_test_split
from deepautoencoder import StackedAutoEncoder
reload(sys)
sys.setdefaultencoding('utf8')



def clip(x):
    return np.clip(x, 1.0, 5.0)

ratingFile = 'ratings.dat'
featureDict = pickle.load(open('featureDict.pkl','rb'))
plotDict = pickle.load(open('feature_plot.pkl','rb'))
categoryDict = pickle.load(open('feature_category.pkl','rb'))
userFeature = pickle.load(open('userFeature.pkl','rb'))
item2vec = pickle.load(open('item2vec.pkl','rb'))
user2vec = pickle.load(open('user2Vec.pkl','rb'))
userIds = set(user2vec.keys())
itemIds = set(item2vec.keys())
items = []
items_plot = []
items_cate = []
# for i in range(1,3953):
# 	try:

# 		#mat = np.array(flatten(featureDict[i][0].tolist()))
# 		#mat = np.array(plotDict[i])
# 		mat = np.array(flatten(categoryDict[i][0].tolist()))
# 		tmp = item2vec[i]
# 		tmp2  = np.hstack((mat,tmp))
# 		items.append(tmp)
# 		items_cate.append(mat)
# 	except Exception,e:
# 		items.append(np.random.normal(0,0.1,300))
# 		items_cate.append(np.random.normal(0,0.1,74))
# 		#items_cate.append(np.random.normal(0,0.1,300))
# items = np.array(items)
# #items_plot = np.array(items_plot)
# items_cate = np.array(items_cate)
# items_cate = items_cate[:,:-1]


# model = StackedAutoEncoder(dims=[50,20], activations=['linear','linear'], epoch=[55000,55000],
#                             loss='rmse', lr=0.001, batch_size=100, print_step=2000)
# items_cate = model.fit_transform(items_cate)

# model2 = StackedAutoEncoder(dims = [256,128], activations = ['linear','linear'],noise ='gaussian',epoch = [25000,25000],
# 							loss='rmse',lr = 0.001,batch_size = 100,print_step =2000)
# items =  model2.fit_transform(items)
# items = np.hstack((items_cate,items))
# print items.shape

# pickle.dump(items,open('itemsEmbedding.pkl','wb'))


items = pickle.load(open('itemsEmbedding.pkl','rb'))

# users = []
# for i in range(1,6041):
# 	try:
# 		users.append(user2vec[i])
# 	except Exception,e:
# 		users.append(np.random.normal(0,0.1,300))
# users = np.array(users)
# userFeature = np.array(userFeature)


# model = StackedAutoEncoder(dims=[15,10], activations=['linear', 'linear'],noise = None, epoch=[55000,25000],
#                             loss='rmse', lr=0.001, batch_size=100, print_step=2000)                         
# usersFeature = model.fit_transform(userFeature)
# model2 = StackedAutoEncoder(dims=[256,128],activations = ['linear','linear'],noise = 'gaussian',epoch = [25000,25000],
# 							loss = 'rmse',lr = 0.001,batch_size = 100,print_step = 2000)
# users = model2.fit_transform(users)
# users = np.hstack((usersFeature,users))
# print users.shape
# pickle.dump(users,open('usersEmbedding.pkl','wb'))

users =pickle.load(open('usersEmbedding.pkl','rb'))




ratings = open(ratingFile,'rb')
r = []
for line in ratings.readlines():
	linelist = line.strip().split('::')
	if len(linelist) != 4:
		print line
	else:
		r.append(linelist)
rating_df = pd.DataFrame(r)
rating_df.columns = ['uid','Iid','score','time']
rating_df['time'] = rating_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(float(x)))
rating_df['uid'] = rating_df['uid'].apply(lambda x:int(x)-1)
rating_df['Iid'] = rating_df['Iid'].apply(lambda x:int(x)-1)
rating_df['score'] = rating_df['score'].apply(lambda x:float(x))
rating_df = rating_df.sort(columns = ['uid','time']).reset_index(drop = True)
train,test = train_test_split(rating_df,test_size = 0.3,random_state = 42)
train = train.reset_index(drop = True)
test = test.reset_index(drop = True)
test = test.iloc[0:20000].reset_index(drop = True)
test_u = test['uid'].values
test_I = list(test['Iid'].values)
test_labels = list(test['score'])
test_user = users[test_u,:]
test_item = items[test_I,:]


reg = 0.05
batch_size = 1000
graph = tf.Graph()





with graph.as_default():
	train_user = tf.placeholder(dtype=  tf.float32,shape=(batch_size,users.shape[1]))
	train_item = tf.placeholder(dtype = tf.float32,shape = (batch_size,items.shape[1]))
	labels = tf.placeholder(dtype = tf.float32,shape = (batch_size,))
	test_user = tf.constant(test_user,dtype = tf.float32)
	test_item = tf.constant(test_item,dtype = tf.float32)
	test_labels = tf.constant(test_labels,dtype = tf.float32)

	weights_item = {
		'weights1':tf.Variable(tf.truncated_normal([items.shape[1],128],stddev = 0.02,mean = 0)),
		'weights2':tf.Variable(tf.truncated_normal([128,64],stddev = 0.02,mean = 0)),
		'weights3':tf.Variable(tf.truncated_normal([64,20],stddev = 0.02,mean = 0)),
		# 'weights4':tf.Variable(tf.truncated_normal([256,100],stddev = np.sqrt(2.0/256))),
	}
	bias_item = {
		'bias1':tf.Variable(tf.zeros([128])),
		'bias2':tf.Variable(tf.zeros([64])),
		'bias3':tf.Variable(tf.zeros([20])),
	}
	


	weights_user = {
		'weights1':tf.Variable(tf.truncated_normal([users.shape[1],128],stddev = 0.02,mean = 0)),
		'weights2':tf.Variable(tf.truncated_normal([128,64],stddev = 0.02,mean = 0)),
		'weights3':tf.Variable(tf.truncated_normal([64,20],stddev = 0.02,mean = 0)),
	}

	bias_user = {
		'bias1':tf.Variable(tf.zeros([128])),
		'bias2':tf.Variable(tf.zeros([64])),
		'bias3':tf.Variable(tf.zeros([20])),
	}

	u_layer1 = tf.nn.relu(tf.matmul(train_user,weights_user['weights1'])+bias_user['bias1'])
	#u_layer1_drop = tf.nn.dropout(u_layer1,0.5)
	u_layer2 = tf.nn.relu(tf.matmul(u_layer1,weights_user['weights2'])+bias_user['bias2'])
	#u_layer2_drop = tf.nn.dropout(u_layer2,0.2)
	u_layer3 = tf.matmul(u_layer2,weights_user['weights3'])+bias_user['bias3']


	i_layer1 = tf.nn.relu(tf.matmul(train_item,weights_item['weights1'])+bias_item['bias1'])
	#i_layer1_drop = tf.nn.dropout(i_layer1,0.5)
	i_layer2 = tf.nn.relu(tf.matmul(i_layer1,weights_item['weights2'])+bias_item['bias2'])
	#i_layer2_drop = tf.nn.dropout(i_layer2,0.2)
	i_layer3 = tf.matmul(i_layer2,weights_item['weights3'])+bias_item['bias3']

	
	infer = tf.reduce_sum(tf.multiply(u_layer3, i_layer3), 1)
	regularizer = tf.add(tf.nn.l2_loss(weights_item['weights1']),
                             tf.nn.l2_loss(weights_user['weights1']))
	regularizer = tf.add(regularizer,tf.nn.l2_loss(weights_item['weights2']))
	regularizer = tf.add(regularizer,tf.nn.l2_loss(weights_item['weights3']))
	regularizer = tf.add(regularizer,tf.nn.l2_loss(weights_user['weights2']))
	regularizer = tf.add(regularizer,tf.nn.l2_loss(weights_user['weights3']))

	global_step = tf.train.get_global_step()
      
	#cost_l2 = tf.reduce_mean(tf.square(tf.subtract(infer,rate_batch)))  
	cost_l2 = tf.nn.l2_loss(tf.subtract(infer, labels))
	penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
	cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
	train_op = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)

	te1_u = tf.nn.relu(tf.matmul(test_user,weights_user['weights1'])+bias_user['bias1'])
	te2_u = tf.nn.relu(tf.matmul(te1_u,weights_user['weights2'])+bias_user['bias2'])
	te3_u = tf.matmul(te2_u,weights_user['weights3'])+bias_user['bias3']
	# te4_u = tf.nn.relu(tf.matmul(te3_u,weights_user['weights4'])+bias_user['bias4'])


	te1_i = tf.nn.relu(tf.matmul(test_item,weights_item['weights1'])+bias_item['bias1'])
	te2_i = tf.nn.relu(tf.matmul(te1_i,weights_item['weights2'])+bias_item['bias2'])
	te3_i =tf.matmul(te2_i,weights_item['weights3'])+bias_item['bias3']

	te_res = tf.reduce_sum(tf.mul(te3_u,te3_i),1)
	test_loss = tf.sqrt(tf.reduce_mean(tf.square(te_res - test_labels)))


epochs = 100
batch_size = 1000
length = len(train)
li = [i for i in range(len(train))]

with tf.Session(graph = graph) as sess:
	sampler_num = len(train) / batch_size
	sess.run(tf.global_variables_initializer())
	errors = deque(maxlen= sampler_num)
	print('Initialized')
	for i in range(epochs):
		print('epcoch %f start') %i
		for j in range(sampler_num):
			slice = random.sample(li,batch_size)
			batch = train.iloc[slice]
			labels_ = list(batch['score'])
			train_user_ = users[batch['uid'].values,:]
			train_item_ = items[batch['Iid'].values,:]
		
			feed_dict ={ train_user: train_user_,train_item:train_item_,labels:labels_}
			_, pred_batch = sess.run([train_op, infer], feed_dict=feed_dict)

			pred_batch = clip(pred_batch)
			errors.append(np.power(pred_batch - np.array(labels_), 2))
		train_err = np.sqrt(np.mean(errors))
		print('train errors %f') % train_err		


