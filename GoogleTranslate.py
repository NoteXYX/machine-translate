import requests  
import json  
import sys  
import urllib  
from bs4 import  BeautifulSoup  
import re  
import execjs  
import os  
import numpy as np
  
  
      
class  Translate:  
    def __init__(self,query_string):  
        self.api_url="https://translate.google.cn"  
        self.query_string=query_string  
        self.headers={  
            "User-Agent":"Mozilla/5.0 (Windows NT 6.1; rv:53.0) Gecko/20100101 Firefox/53.0"  
                      }  
          
    def get_url_param_data(self):  
        url_param_part=self.api_url+"/translate_a/single?"  
        url_param=url_param_part+"client=t&sl=en&tl=es&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&source=btn&ssel=3&tsel=3&kc=0&"  
        #sl为源语言，tl为目标语言
        url_get=url_param+"tk="+str(self.get_tk())+"&q="+str(self.get_query_string())  
        #print(url_get)  
        return  url_get  
      
    def get_query_string(self):  
        query_url_trans=urllib.parse.quote(self.query_string)#汉字url编码  
        return  query_url_trans  
       
    def get_tkk(self):  
        part_jscode_2="\n"+"return TKK;"  
        tkk_page=requests.get(self.api_url,headers=self.headers)  
        tkk_code=BeautifulSoup(tkk_page.content,'lxml')  
        patter= re.compile(r'(TKK.*?\);)', re.I | re.M)  
        part_jscode=re.findall(patter,str(tkk_code))  
        #print(part_jscode[0])  
        js_code=part_jscode[0]+part_jscode_2  
        with open ("googletranslate.js","w")  as  f:  
            f.write(js_code)  
            f.close  
        tkk_value=execjs.compile(open(r"googletranslate.js").read()).call('eval')  
        #print(tkk_value)  
        return tkk_value  
      
    def get_tk(self):  
        tk_value=execjs.compile(open(r"googletranslate_1.js").read()).call('tk',self.query_string,self.get_tkk())  
        #print(tk_value)  
        return tk_value  
            
      
    def parse_url(self):  
        response=requests.get(self.get_url_param_data(),headers=self.headers)  
        return response.content.decode()  
      
      
    def  get_trans_ret(self,json_response):  
        dict_response=json.loads(json_response)  
        ret=dict_response[0][0][0]  
        #print(ret) 
        return ret 
          
          
    def  run(self):  
        json_response=self.parse_url()  
        n=self.get_trans_ret(json_response)
        return n  
         
         
if  __name__=="__main__":
    vocab = np.load('vocab/vocabEN-ES.npy')
    vocab_array=np.array(['',''])
    for i in range(1116):
        google=Translate(vocab[i][0].lower())
        row=np.array([vocab[i][0],google.run().lower()])
        vocab_array=np.row_stack((vocab_array,row))
        print('谷歌翻译第%d'%(i+1) + '个单词完成！')
    vocab_array=np.delete(vocab_array,0,0)
    np.save("123.npy", vocab_array)
    a=np.load("123.npy")
    print(a)
    print(a.shape)
    '''f_en = open('test_removed.txt', 'r')
    mystr = f_en.read()
    en_list = mystr.split()
    vocab_array=np.array(['',''])
    for i in range(1280):
        google=Translate(en_list[i])
        row=np.array([en_list[i],google.run().lower()])
        vocab_array=np.row_stack((vocab_array,row))
        print('谷歌翻译第%d'%(i+1)+'个单词完成！')

    vocab_array = np.delete(vocab_array,0,0)
    print(vocab_array)
    np.save("test1000EN-FR.npy", vocab_array)
    f_en.close()'''
