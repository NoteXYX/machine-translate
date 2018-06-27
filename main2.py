import gensim
import numpy as np

model_EN = gensim.models.Word2Vec.load("../v6_FR_SG/v6_FR_SG_800.model")
model_FR = gensim.models.Word2Vec.load("../v6_EN_SG/v6_EN_300.model")
vocab = np.load("GT/testFR-EN.npy")
new = np.array(['',''])
#Thta = np.zeros((400,400))
#Alpha = 0.003
num = vocab.shape[0]
for i in range(num):	   #循环每个翻译对
	word_EN = vocab[i][0]
	word_FR = vocab[i][1]
	#vec_EN = model_EN.wv[word_EN]
	#vec_FR = model_FR.wv[word_FR]
	try:
		vec_EN = model_EN.wv[word_EN]
		vec_FR = model_FR.wv[word_FR]
	except Exception as e:
		#np.delete(vocab,i,0)
		#np.save("GT/NEWvocabEN-ZH.npy",vocab)
		#print("测试第%d个翻译对"%(i))
		continue
	else:
		row = np.array([word_EN , word_FR])
		new=np.row_stack((new,row))
		continue
		#print("测试第%d个翻译对"%(i))
	#vec_EN.shape = (1,400)	#将词向量转置成1*400的，原来是400*1的
	#vec_ES.shape = (1,400)	#将词向量转置成1*400的，原来是400*1的
	#print("测试第%d个翻译对"%(i))
	
new = np.delete(new,0,0)
np.save("GT/test1000FR-EN.npy",new)
vocab1 = np.load("GT/test1000FR-EN.npy")
print(vocab1.shape)


