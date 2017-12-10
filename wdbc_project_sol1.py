# WDBC dataset Project: 569 instances 32 features.
# output 2 means benign(유방암 x), 4 means Malignant(유방암 o)
# Solution 1 ( MLP)


# Lab 13 Tensorboard
import tensorflow as tf
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
#tf.set_random_seed(777)  # reproducibility
# wbc = np.loadtxt('Data/breast-cancer-wisconsin.data.txt',delimiter=',',dtype=np.float32)
# pd.set_option('display.width',1000) # 1000개의 글자까지는 수평으로 출력하겠다. defaul값 변경
# df = pd.read_csv('Data/breast-cancer-wisconsin.data.txt',header=None,sep=",",engine='python',names=['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])
# print(df)
# print(df[df['Bare Nuclei'] != "?"].mean())
#

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
# data.append(wbc[])

#print(wbc[1:])
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
# parameters
learning_rate = 0.001
training_epochs = 45
batch_size = 20

# input place holders
X = tf.placeholder(tf.float32, [None, 30])
Y = tf.placeholder(tf.float32, [None, 2])

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)
# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
with tf.variable_scope('layer1') as scope:
    W1 = tf.get_variable("W", shape=[30, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([20]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    tf.summary.histogram("X", X)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("bias", b1)
    tf.summary.histogram("layer", L1)
with tf.variable_scope('layer2') as scope:
    W2 = tf.get_variable("W", shape=[20, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([20]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("bias", b2)
    tf.summary.histogram("layer", L2)
with tf.variable_scope('layer3') as scope:
    W3 = tf.get_variable("W", shape=[20, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([20]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    tf.summary.histogram("weights", W3)
    tf.summary.histogram("bias", b3)
    tf.summary.histogram("layer", L3)
with tf.variable_scope('layer4') as scope:
    W4 = tf.get_variable("W", shape=[20, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([20]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    tf.summary.histogram("weights", W4)
    tf.summary.histogram("bias", b4)
    tf.summary.histogram("layer", L4)
with tf.variable_scope('layer5') as scope:
    W5 = tf.get_variable("W", shape=[20, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([20]))
    L5 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L5 = tf.nn.dropout(L4, keep_prob=keep_prob)
    tf.summary.histogram("weights", W5)
    tf.summary.histogram("bias", b5)
    tf.summary.histogram("layer", L5)
with tf.variable_scope('layer6') as scope:
    W6 = tf.get_variable("W", shape=[20, 2],
                         initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([2]))
    hypothesis = tf.matmul(L5, W6) + b6
    tf.summary.histogram("weights", W6)
    tf.summary.histogram("bias", b6)
    tf.summary.histogram("hypothesis", hypothesis)
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.summary.scalar("loss", cost)
# Summary
summary = tf.summary.merge_all()
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create summary writer
writer = tf.summary.FileWriter('./logs/wdbc_project_sol1_2')
writer.add_graph(sess.graph)
global_step = 0
print('Start learning!')
# train my model
print(training_data_x[0:120].shape)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(training_data_x.__len__() / batch_size)

    batch_i = 0
    for i in range(total_batch):
        batch_xs = training_data_x[batch_i:batch_i+20]
        batch_ys = training_data_y_refined[batch_i:batch_i+20]
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
        writer.add_summary(s, global_step=global_step)
        global_step += 1
        avg_cost += sess.run(cost, feed_dict=feed_dict) / total_batch
        batch_i = batch_i + 20
        if(batch_i > training_data_x.__len__()):
            temp = training_data_x.__len__()- batch_i + 20
            batch_i = batch_i - 20 + temp


    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')
# Test model and check accuracy

# P: 유방암이라고 예측. N: 유방암이 아니라고 예측
TP = tf.count_nonzero(tf.argmax(hypothesis,1)*tf.argmax(Y,1),dtype=tf.float32) # 왜 이게 될까? -> element wise하게 곱하기 때문에, 예측 ,label 둘다 1인 경우만
TN = tf.count_nonzero((tf.argmax(hypothesis,1)-1)*(tf.argmax(Y,1)-1),dtype=tf.float32)
FP = tf.count_nonzero(tf.argmax(hypothesis,1)*(tf.argmax(Y,1)-1),dtype=tf.float32)
FN = tf.count_nonzero((tf.argmax(hypothesis,1)-1)*tf.argmax(Y,1),dtype=tf.float32)


precision = tf.divide(TP,TP+FP) # 내가 유방암이라고 예측한 사람 중에 맞춘 비율
recall = tf.divide(TP,TP+FN) # 실제로 유방암인 사람중에 유방암이라고 예측한 비율
specificity = tf.divide(TN,TN+FP) # 실제로 정상인 사람중에 내가 정상이라고 예측한 비율





print('Precision: ', sess.run(precision, feed_dict={X: test_data_x,Y: test_data_y_refined,keep_prob: 1}))
print('Recall: ', sess.run(recall, feed_dict={X: test_data_x,Y: test_data_y_refined,keep_prob: 1}))
print('specificity: ', sess.run(specificity, feed_dict={X: test_data_x,Y: test_data_y_refined,keep_prob: 1}))
print('TP: ', sess.run(TP, feed_dict={X: test_data_x,Y: test_data_y_refined,keep_prob: 1}))
print('TN: ', sess.run(TN, feed_dict={X: test_data_x,Y: test_data_y_refined,keep_prob: 1}))
print('FP: ', sess.run(FP, feed_dict={X: test_data_x,Y: test_data_y_refined,keep_prob: 1}))
print('FN: ', sess.run(FN, feed_dict={X: test_data_x,Y: test_data_y_refined,keep_prob: 1}))


#print("Precision: ", sess.run(precision,feed_dict={X: test_data_x, Y: test_data_y_refined,keep_prob: 1}))
# # Get one and predict
r = random.randint(0,100)
#
if(test_data_y_refined[r,0]==1):
    print("Label: 유방암x")
elif(test_data_y_refined[r,-1]==1):
    print("Label: 유방암")

print("Prediction: ", sess.run(tf.argmax(hypothesis,1) , feed_dict={X: test_data_x[r].reshape(1,-1), keep_prob: 1}))
