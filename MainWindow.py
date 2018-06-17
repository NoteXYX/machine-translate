import sys
from PyQt5.QtCore import  Qt
from PyQt5.QtWidgets import *
import gensim
import numpy as np
#from GT import Translate

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('基于语言相似性的机器翻译')
        self.setGeometry(400, 200, 500, 350)
        self.label_source = QLabel(self)
        self.label_source.setText("源语言")
        self.label_source.move(80,20)
        self.label_target = QLabel(self)
        self.label_target.setText("目标语言")
        self.label_target.move(320,20)
        self.combo_source = QComboBox(self)
        self.combo_source.addItem("英文")
        self.combo_source.addItem("西班牙文")
        self.combo_source.addItem("法文")
        self.combo_source.move(50, 50)
        self.combo_target = QComboBox(self)
        self.combo_target.addItem("西班牙文")
        self.combo_target.addItem("法文")
        self.combo_target.addItem("英文")
        self.combo_target.move(300, 50)
        self.source_line = QLineEdit(self)
        self.source_line.move(50,150)
        self.target_line = QTextEdit(self)
        self.target_line.setGeometry(300, 140, 150, 60)
        self.rd1 = QRadioButton("SGD",self)
        self.rd1.move(320, 210)
        self.rd2 = QRadioButton("Adam",self)
        self.rd2.move(380,210)
        self.rd1.toggled.connect(self.choose)
        self.rd2.toggled.connect(self.choose)
        self.rd1.setChecked(True)
        self.button_exchange = QPushButton(self)
        self.button_exchange.setGeometry(200,45,60,30)
        self.button_exchange.clicked.connect(self.exchange)
        self.button_exchange.setStyleSheet('QPushButton{border-image:url(22.jpg)}')
        '''self.target_line = QLineEdit(self)
        self.target_line.move(300,150)'''
        '''self.label = QLabel(self)
        self.label.move(500,50)
        self.label.setText("谷歌翻译结果")
        self.GT_line = QLineEdit(self)
        self.GT_line.move(500, 150)'''
        self.button_translate  = QPushButton(self)
        self.button_translate.setText('翻译')
        self.button_translate.setGeometry(200,270,100,40)
        self.button_translate.clicked.connect(self.translate)

    def choose(self):
        if self.rd1.isChecked():
            self.ThtaEN_ES = np.load("Thta/ThtaEN-ES0.01_6000.npy")
            self.ThtaEN_FR = np.load("Thta/ThtaEN-FR0.02_6000.npy")
            self.ThtaES_EN = np.load("Thta/ThtaES-EN0.02_5000.npy")
            self.ThtaFR_EN = np.load("Thta/ThtaFR-EN0.02_5000.npy")
            self.ThtaES_FR = np.load("Thta/ThtaES-FR0.01_6000.npy")
            self.ThtaFR_ES = np.load("Thta/ThtaFR-ES0.01_6000.npy")
        if self.rd2.isChecked():
            self.ThtaEN_ES = np.load("Thta/Adam_ThtaEN-ES0.0008_5000.npy")
            self.ThtaEN_FR = np.load("Thta/Adam_ThtaEN-FR0.001_5000.npy")
            self.ThtaES_EN = np.load("Thta/Adam_ThtaES-EN0.001_5000.npy")
            self.ThtaFR_EN = np.load("Thta/Adam_ThtaFR-EN0.001_5000.npy")
            self.ThtaES_FR = np.load("Thta/Adam_ThtaES-FR0.001_5000.npy")
            self.ThtaFR_ES = np.load("Thta/Adam_ThtaFR-ES0.001_5000.npy")

    def warning(self):  # 消息：警告
        QMessageBox.information(self, "Error", self.tr("输入的源词不在模型中！"))

    def translate(self):
        if self.combo_source.currentText() == "英文" and self.combo_target.currentText() == "西班牙文":
            self.word_EN = self.source_line.text()
            try:
                self.vec_EN = model_EN_800.wv[self.word_EN]
            except Exception as e:
                self.warning()
            else:
                self.vec_EN.shape = (1, 800)
                self.result_vec = np.dot(self.vec_EN, self.ThtaEN_ES)
                self.result_vec.shape = (200,)
                self.result_word = model_ES_200.wv.similar_by_vector(self.result_vec, topn=3, restrict_vocab=None)
                self.target_line.clear()
                for result_num in range(3):
                    self.target_line.append(self.result_word[result_num][0])

                '''self.google = Translate(self.word_EN)
                self.google.sl = "en"
                self.google.tl = "es"
                self.GT_line.setText(self.google.run())'''

        if self.combo_source.currentText() == "英文" and self.combo_target.currentText() == "法文":
            self.word_EN = self.source_line.text()
            try:
                self.vec_EN = model_EN_800.wv[self.word_EN]
            except Exception as e:
                self.warning()
            else:
                self.vec_EN.shape = (1, 800)
                self.result_vec = np.dot(self.vec_EN, self.ThtaEN_FR)
                self.result_vec.shape = (200,)
                self.result_word = model_FR_200.wv.similar_by_vector(self.result_vec, topn=3, restrict_vocab=None)
                self.target_line.clear()
                for result_num in range(3):
                    self.target_line.append(self.result_word[result_num][0])


        if self.combo_source.currentText() == "西班牙文" and self.combo_target.currentText() == "英文":
            self.word_ES = self.source_line.text()
            try:
                self.vec_ES = model_ES_800.wv[self.word_ES]
            except Exception as e:
                self.warning()
            else:
                self.vec_ES.shape = (1, 800)
                self.result_vec = np.dot(self.vec_ES, self.ThtaES_EN)
                self.result_vec.shape = (300,)
                self.result_word = model_EN_300.wv.similar_by_vector(self.result_vec, topn=3, restrict_vocab=None)
                self.target_line.clear()
                for result_num in range(3):
                    self.target_line.append(self.result_word[result_num][0])

        if self.combo_source.currentText() == "法文" and self.combo_target.currentText() == "英文":
            self.word_FR = self.source_line.text()
            try:
                self.vec_FR = model_FR_800.wv[self.word_FR]
            except Exception as e:
                self.warning()
            else:
                self.vec_FR.shape = (1, 800)
                self.result_vec = np.dot(self.vec_FR,self.ThtaFR_EN)
                self.result_vec.shape = (300,)
                self.result_word = model_EN_300.wv.similar_by_vector(self.result_vec, topn=3, restrict_vocab=None)
                self.target_line.clear()
                for result_num in range(3):
                    self.target_line.append(self.result_word[result_num][0])

        if self.combo_source.currentText() == "西班牙文" and self.combo_target.currentText() == "法文":
            self.word_ES = self.source_line.text()
            try:
                self.vec_ES = model_ES_800.wv[self.word_ES]
            except Exception as e:
                self.warning()
            else:
                self.vec_ES.shape = (1, 800)
                self.result_vec = np.dot(self.vec_ES, self.ThtaES_FR)
                self.result_vec.shape = (200,)
                self.result_word = model_FR_200.wv.similar_by_vector(self.result_vec, topn=3, restrict_vocab=None)
                self.target_line.clear()
                for result_num in range(3):
                    self.target_line.append(self.result_word[result_num][0])

        if self.combo_source.currentText() == "法文" and self.combo_target.currentText() == "西班牙文":
            self.word_FR = self.source_line.text()
            try:
                self.vec_FR = model_FR_800.wv[self.word_FR]
            except Exception as e:
                self.warning()
            else:
                self.vec_FR.shape = (1, 800)
                self.result_vec = np.dot(self.vec_FR, self.ThtaFR_ES)
                self.result_vec.shape = (200,)
                self.result_word = model_ES_200.wv.similar_by_vector(self.result_vec, topn=3, restrict_vocab=None)
                self.target_line.clear()
                for result_num in range(3):
                    self.target_line.append(self.result_word[result_num][0])

    def exchange(self):
        self.current_index = self.combo_source.currentIndex()
        self.combo_source.setCurrentIndex((self.combo_target.currentIndex() + 1) % 3)
        self.combo_target.setCurrentIndex((self.current_index + 2) % 3)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    model_EN_800 = gensim.models.Word2Vec.load("model/v6_EN_SG_800.model")
    model_EN_300 = gensim.models.Word2Vec.load("model/v6_EN_SG_300.model")
    model_ES_200 = gensim.models.Word2Vec.load("model/v6_ES_SG_200.model")
    model_ES_800 = gensim.models.Word2Vec.load("model/v6_ES_SG_800.model")
    model_FR_200 = gensim.models.Word2Vec.load("model/v6_FR_SG_200.model")
    model_FR_800 = gensim.models.Word2Vec.load("model/v6_FR_SG_800.model")

    

    sys.exit(app.exec_())
