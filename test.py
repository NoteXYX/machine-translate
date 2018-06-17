import gensim
import numpy as np
import xlwt
model_EN = gensim.models.Word2Vec.load("model/v6_EN_SG_800.model")
model_FR = gensim.models.Word2Vec.load("model/v6_ES_SG_200.model")
#workbook = xlwt.Workbook(encoding = 'utf-8')
#worksheet = workbook.add_sheet('Result')
Thta = np.load("Thta/ThtaEN-ES0.01_4000.npy")
test = np.load("test/test1000EN-ES.npy")
'''font1 = xlwt.Font()
font1.height=0x00E8
font1.name = '宋体'
style1 = xlwt.XFStyle()
style1.font = font1
worksheet.write(0, 0, label = '英文测试单词', style = style1)
worksheet.col(0).width = 3333
worksheet.write(0, 1, label = '预测的西班牙语译文', style = style1)
worksheet.col(1).width = 4000
worksheet.write(0, 2, label = '词典给出的西班牙语译文', style = style1)
worksheet.col(2).width = 4400
worksheet.write(0, 3, label = '对错', style = style1)
worksheet.col(3).width = 4400'''
num = 0
true_Word=0.0
while num < 1000:
	word_EN = test[num][0]
	word_FR = test[num][1]
	vec_Test = model_EN.wv[word_EN]
	vec_Test.shape = (1,800)
	b = np.dot(vec_Test,Thta)
	b.shape = (200,)
	e = model_FR.wv.similar_by_vector(b, topn=5, restrict_vocab=None)
	print(e[0][0])
	#worksheet.write(num+1, 0, label = word_EN)
	#worksheet.write(num+1, 1, label = [e[k][0]+'  ' for k in range(1)])
	#worksheet.write(num+1, 2, label = word_FR)
	tmp=5
	for i in range(tmp):
		if e[i][0] == word_FR:
			#worksheet.write(num+1, 3, label = '✔️')
			true_Word+=1
			break
		#else:
			#worksheet.write(num+1, 3, label = '×')
			#continue
		elif i == (tmp-1):
			#worksheet.write(num+1, 3, label = '×')
			break
	print('测试完成%d个单词'%(num+1))
	num += 1

#worksheet.write(num+1, 0, label = '正确率', style = style1)
#worksheet.write(num+1, 1, label = str(true_Word/num*100)+'%')
print(str(true_Word/num*100)+'%')
#workbook.save('result/Adam_ThtaEN-ES0.0009_4000.npy@1.xls')