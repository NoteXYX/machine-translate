import tensorflow as tf
import os
import logging
import sys
import gensim
import numpy as np
import random
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
starttime = datetime.datetime.now()
# Model parameters
W = tf.Variable(tf.zeros([800,200]))

# Model input and output
x = tf.placeholder(tf.float32, shape=[1, 800])
linear_model = tf.matmul(x,W)
y = tf.placeholder(tf.float32, shape=[1, 200])

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

# training data
model_EN_800 = gensim.models.Word2Vec.load("model/v6_EN_SG_800.model")
model_ES_200 = gensim.models.Word2Vec.load("model/v6_ES_SG_200.model")
vocab = np.load("vocab/vocabEN-ES.npy")

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
for iters in range(3000):
    i = random.randint(0, (vocab.shape[0] - 1))
    word_EN = vocab[i][0]
    word_ES = vocab[i][1]
    x_train = model_EN_800.wv[word_EN]
    y_train = model_ES_200.wv[word_ES]
    x_train.shape = [1,800]
    y_train.shape = [1,200]
    sess.run(train, {x: x_train, y: y_train})
    logger.info("训练%d个翻译对" % (iters))

# evaluate training accuracy
curr_W, curr_loss = sess.run([W, loss], {x: x_train, y: y_train})
print("W: %s loss: %s" % (curr_W, curr_loss))
print(W.shape)
np.save("Thta/TF_ThtaEN-ES0.02_3000.npy",curr_W)
endtime = datetime.datetime.now()
print ('共运行' + str((endtime - starttime).seconds) + '秒')
#saver = tf.train.Saver()


