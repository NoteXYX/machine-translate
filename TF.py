import tensorflow as tf
import os
import logging
import sys
import gensim
import numpy as np
import random
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ' '.join(sys.argv))
	# Model parameters
	starttime = datetime.datetime.now()
	W = tf.Variable(tf.zeros([800,200]))
	# Model input and output
	x = tf.placeholder(tf.float32, shape=[1, 800])
	linear_model = tf.matmul(x,W)
	y = tf.placeholder(tf.float32, shape=[1, 200])
	# loss
	loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
	# optimizer
	optimizer = tf.train.GradientDescentOptimizer(0.009)
	train = optimizer.minimize(loss)
	# training data
	model_EN_800 = gensim.models.Word2Vec.load("../v6_EN_SG/v6_EN_SG_800.model")
	model_ES_200 = gensim.models.Word2Vec.load("../v6_ES_SG/v6_ES_SG_200.model")
	vocab = np.load("GT/vocabEN-ES.npy")
	# training loop
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)  # reset values to wrong
	for iters in range(6000):
		i = random.randint(0, 1854)
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
	print(curr_W.shape)
	endtime = datetime.datetime.now()
	print ('共运行' + str((endtime - starttime).seconds) + '秒')
	np.save("GT/ThtaEN-ES/TF/ThtaEN-ES0.009_6000.npy",curr_W)



