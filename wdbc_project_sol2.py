# WDBC dataset Project: 569 instances 32 features.
# output 2 means benign(유방암 x), 4 means Malignant(유방암 o)

# Solution 2 (CNN - conv1d)

import tensorflow as tf
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data

wbc = np.loadtxt('Data/wdbc_dataset.csv',delimiter=',',dtype=np.float32)
np.random.shuffle(wbc)
training_data_x = wbc[:-100,2:] # training_data set X
training_data_y = wbc[:-100,1] # training_Data set Y
training_data_y_refined = np.zeros((training_data_y.__len__(),2))
for i in range(training_data_y.__len__()):
    if(training_data_y[i]==2): # 유방암이 아닌 경우
        training_data_y_refined[i,0] = 1
    else:
        training_data_y_refined[i,-1] = 1
print(training_data_y_refined) # shape은 663X2.
print('--------------------------')
test_data_x = wbc[-100:,2:] # test_data set_x
test_data_y = wbc[-100:,1] # test_data_set_y
test_data_y_refined = np.zeros((test_data_y.__len__(),2))
for i in range(test_data_y.__len__()):
    if(test_data_y[i]==2): # 유방암이 아닌 경우
        test_data_y_refined[i,0] = 1
    else:
        test_data_y_refined[i,-1] = 1
print(test_data_y_refined)


learning_rate = 0.001
training_epochs = 30
batch_size = 20

class Model:
    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):

            self.training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None,30])
            x_in = tf.reshape(self.X,[-1,30,1])
            self.Y = tf.placeholder(tf.float32,[None,2])


            conv1 = tf.layers.conv1d(inputs=x_in, kernel_size=3, filters=30, padding="SAME", activation=tf.nn.relu)
            # conv1 후에 [None,30,30] 로 나뉘어짐
            pool1 = tf.layers.max_pooling1d(inputs=conv1,pool_size=2,padding="SAME",strides=2)
            # pool1 후에 [None,15,30]
            dropout1 = tf.layers.dropout(inputs=pool1,rate=0.7,training=self.training)



            conv2 = tf.layers.conv1d(inputs=dropout1,filters=60,kernel_size=3,padding="SAME",activation=tf.nn.relu)
            # conv2 후에 [None,15,60]
            pool2 = tf.layers.max_pooling1d(inputs=conv2,pool_size=2,padding="SAME",strides=2)
            # pool2 후에 [None,8,60]
            dropout2 = tf.layers.dropout(inputs=pool2,rate=0.7,training=self.training)


            conv3 = tf.layers.conv1d(inputs=dropout2, filters=60, kernel_size=3, padding="SAME", activation=tf.nn.relu)
            # conv2 후에 [None,8,60]
            pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, padding="SAME", strides=2)
            # pool2 후에 [None,4,60]
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)




            flat = tf.reshape(dropout3,[-1,4*60])
            dense4 = tf.layers.dense(inputs=flat,units=120,activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,rate=0.7,training=self.training)



            self.logits = tf.layers.dense(inputs=dropout4,units=2)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_prediction= tf.equal(tf.argmax(self.logits,1),tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        # P: 유방암이라고 예측. N: 유방암이 아니라고 예측
        TP = tf.count_nonzero(tf.argmax(self.logits, 1) * tf.argmax(self.Y, 1),
                              dtype=tf.float32)  # 왜 이게 될까? -> element wise하게 곱하기 때문에, 예측 ,label 둘다 1인 경우만
        TN = tf.count_nonzero((tf.argmax(self.logits, 1) - 1) * (tf.argmax(self.Y, 1) - 1), dtype=tf.float32)
        FP = tf.count_nonzero(tf.argmax(self.logits, 1) * (tf.argmax(self.Y, 1) - 1), dtype=tf.float32)
        FN = tf.count_nonzero((tf.argmax(self.logits, 1) - 1) * tf.argmax(self.Y, 1), dtype=tf.float32)

        self.precision = tf.divide(TP, TP + FP)  # 내가 유방암이라고 예측한 사람 중에 맞춘 비율
        self.recall = tf.divide(TP, TP + FN)  # 실제로 유방암인 사람중에 유방암이라고 예측한 비율
        self.specificity = tf.divide(TN, TN + FP)  # 실제로 정상인 사람중에 내가 정상이라고 예측한 비율

    def predict(self,x_test,training=False):
        return self.sess.run(self.logits,feed_dict={self.X: x_test,self.training: training})

    def get_accuracy(self,x_test,y_test,training=False):
        return self.sess.run(self.accuracy,feed_dict={self.X: x_test,self.Y: y_test,self.training: training})
    def get_precision(self,x_test,y_test,training=False):
        return self.sess.run(self.precision,feed_dict={self.X: x_test,self.Y: y_test,self.training: training})
    def get_recall(self,x_test,y_test,training=False):
        return self.sess.run(self.recall,feed_dict={self.X: x_test,self.Y: y_test,self.training: training})
    def get_specificity(self,x_test,y_test,training=False):
        return self.sess.run(self.specificity,feed_dict={self.X: x_test,self.Y: y_test,self.training: training})
    def train(self,x_data,y_data,training=True):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X: x_data,self.Y: y_data,self.training: training})

''' single model accuracy'''
'''
sess = tf.Session()

m1 = Model(sess,"m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')


for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(training_data_x.__len__() / batch_size)

    batch_i = 0
    for i in range(total_batch):
        batch_xs = training_data_x[batch_i:batch_i+20]
        batch_ys = training_data_y_refined[batch_i:batch_i+20]
        c, _ = m1.train(batch_xs,batch_ys)
        avg_cost += c / total_batch
        batch_i = batch_i + 20
        if(batch_i > training_data_x.__len__()):
            temp = training_data_x.__len__()- batch_i + 20
            batch_i = batch_i - 20 + temp


    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')
print('Accuracy:', m1.get_accuracy(test_data_x,test_data_y_refined))

'''
sess = tf.Session()

models = []
num_models = 10
for m in range(num_models):
    models.append(Model(sess,"model"+str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')


for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(training_data_x.__len__() / batch_size)

    batch_i = 0
    for i in range(total_batch):
        batch_xs = training_data_x[batch_i:batch_i+20]
        batch_ys = training_data_y_refined[batch_i:batch_i+20]
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs,batch_ys)
            avg_cost_list[m_idx] += c/ total_batch

        batch_i = batch_i + 20
        if(batch_i > training_data_x.__len__()):
            temp = training_data_x.__len__()- batch_i + 20
            batch_i = batch_i - 20 + temp
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
print('Learning Finished!')



test_size = len(test_data_y_refined)
predictions = np.zeros([test_size,2])
for m_idx,m in enumerate(models):
    print(m_idx,'Precision: ',m.get_precision(test_data_x,test_data_y_refined),'recall: ',m.get_recall(test_data_x,test_data_y_refined),'Specificity: ',m.get_specificity(test_data_x,test_data_y_refined))

    p = m.predict(test_data_x)
    predictions += p

# P: 유방암이라고 예측. N: 유방암이 아니라고 예측
ensemble_TP = tf.count_nonzero(tf.argmax(predictions,1)*tf.argmax(test_data_y_refined,1),dtype=tf.float32) # 왜 이게 될까? -> element wise하게 곱하기 때문에, 예측 ,label 둘다 1인 경우만
ensemble_TN = tf.count_nonzero((tf.argmax(predictions,1)-1)*(tf.argmax(test_data_y_refined,1)-1),dtype=tf.float32)
ensemble_FP = tf.count_nonzero(tf.argmax(predictions,1)*(tf.argmax(test_data_y_refined,1)-1),dtype=tf.float32)
ensemble_FN = tf.count_nonzero((tf.argmax(predictions,1)-1)*tf.argmax(test_data_y_refined,1),dtype=tf.float32)


ensemble_precision = tf.divide(ensemble_TP,ensemble_TP+ensemble_FP) # 내가 유방암이라고 예측한 사람 중에 맞춘 비율
ensemble_recall = tf.divide(ensemble_TP,ensemble_TP+ensemble_FN) # 실제로 유방암인 사람중에 유방암이라고 예측한 비율
ensemble_specificity = tf.divide(ensemble_TN,ensemble_TN+ensemble_FP) # 실제로 정상인 사람중에 내가 정상이라고 예측한 비율


print('Ensemble precision: ',sess.run(ensemble_precision))
print('Ensemble recall: ',sess.run(ensemble_recall))
print('Ensemble specificity : ',sess.run(ensemble_specificity))

