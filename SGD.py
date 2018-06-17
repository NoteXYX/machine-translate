import gensim
import numpy as np
import random
import logging
import os
import sys
import multiprocessing
import datetime
if __name__ == '__main__':
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ' '.join(sys.argv))
	starttime = datetime.datetime.now()
	model_EN = gensim.models.Word2Vec.load("model/v6_EN_SG_800.model")
	model_FR = gensim.models.Word2Vec.load("model/v6_ES_SG_200.model")
	vocab = np.load("vocab/vocabEN-ES.npy")  #1776
	Thta = np.zeros((800,200))
	Alpha = 0.01
	#Eps = 0.00001
	iters = 1
	#loss = 50
	while iters <= 4000:
		i = random.randint(0,(vocab.shape[0] - 1))
		word_EN = vocab[i][0]
		word_FR = vocab[i][1]
		vec_EN = model_EN.wv[word_EN]
		vec_FR = model_FR.wv[word_FR]
		vec_EN.shape = (1,800)	#将词向量转置成1*800的，原来是800*1的
		vec_FR.shape = (1,200)	#将词向量转置成1*200的，原来是200*1的
		H = np.zeros((1,200))
		H = np.dot(vec_EN, Thta)
		loss = H - vec_FR
		#result = np.sum(Z * Z)**(1/2)
		for j in range(200):
			for m in range(800):
				Thta[m][j] = Thta[m][j] - Alpha * loss[0][j] * vec_EN[0][m] #将该列的每个Thta更新一下

		logger.info("训练%d个翻译对" % (iters))
		iters = iters + 1

	endtime = datetime.datetime.now()
	print ('共运行' + str((endtime - starttime).seconds) + '秒')
	print(Thta)
	print(Thta.shape)
	print(Thta.size)
	np.save("Thta/ThtaEN-ES0.01_4000.npy",Thta)
	
