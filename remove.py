#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import os.path
import six
import sys
import re
import codecs

#reload(sys)
#sys.setdefaultencoding('utf8')
if __name__ == '__main__':
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ' '.join(sys.argv))
	if len(sys.argv) != 3:
		print("Using: python remove.py xxx.txt xxxx.txt")
		sys.exit(1)
	inp, outp = sys.argv[1:3]
	output = codecs.open(outp, 'w', encoding='utf-8')
	inp = codecs.open(inp, 'r', encoding='utf-8')
	i = 0
	for line in inp.readlines():
		ss = re.sub("[\s+\.\!\/_,;-><¿#&«-»=|1234567890¡?():$%^*(+\"\']+|[+——！，。？、【】《》“”‘’~@#￥%……&*（）''""]+".decode("utf-8"), " ".decode("utf-8"),line)
		ss+="\n"
		output.write("".join(ss.lower()))
		i=i+1
		if (i % 10000 == 0):
			logger.info("Saved " + str(i) + " articles")
			#break
	output.close()
	inp.close()
	logger.info("Finished removed words!")