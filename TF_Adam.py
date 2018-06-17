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
W = tf.Variable(tf.zeros([200,300]))

# Model input and output
x = tf.placeholder(tf.float32, shape=[1, 200])
linear_model = tf.matmul(x,W)
y = tf.placeholder(tf.float32, shape=[1, 300])

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0009,beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
train = optimizer.minimize(loss)

# training data
model_EN_800 = gensim.models.Word2Vec.load("model/wiki_ZH.model")
model_ES_200 = gensim.models.Word2Vec.load("model/v6_EN_SG_300.model")
vocab = np.load("vocab/vocabEN-ZH.npy")

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
for iters in range(4000):
    i = random.randint(0, (vocab.shape[0] - 1))
    word_EN = vocab[i][1]
    word_ES = vocab[i][0]
    x_train = model_EN_800.wv[word_EN]
    y_train = model_ES_200.wv[word_ES]
    x_train.shape = [1,200]
    y_train.shape = [1,300]
    sess.run(train, {x: x_train, y: y_train})
    logger.info("训练%d个翻译对" % (iters))
    #logger.info('Learning rate: %s' % (sess.run(optimizer._lr)))

# evaluate training accuracy
curr_W, curr_loss = sess.run([W, loss], {x: x_train, y: y_train})
print("W: %s loss: %s" % (curr_W, curr_loss))
print(W.shape)
np.save("Thta/TF_Adam_ZH-EN0.0009_4000.npy",curr_W)
endtime = datetime.datetime.now()
print ('共运行' + str((endtime - starttime).seconds) + '秒')
#saver = tf.train.Saver()


