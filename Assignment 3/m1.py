import nltk
nltk.download('punkt')

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from bs4 import BeautifulSoup
import os
import json
import time

class Indexer:
	def __init__(self, dataset_path, index_file, lookup_table_file, debug):
		self.dataset_path = dataset_path
		self.index_file = index_file
		self.lookup_table_file = lookup_table_file
		self.debug = debug

		self.ss = SnowballStemmer(language="english")
		self.doc_id = 0
		self.doc_id_to_url = {}
		self.inverted_index = defaultdict(dict)

	def start(self):
		if debug: 
			start = time.perf_counter()
			print("START - Starting program...")

		for subdir, dirs, files in os.walk(self.dataset_path):
			for name in files:
				self.process(subdir + os.sep + name)
		self.dump()

		if debug:
			end = time.perf_counter()
			print("START - Total number of documents:{}".format(self.doc_id))
			print("START - Total Number of tokens: {}".format(len(self.inverted_index)))
			print("START - Program execution time: {} seconds".format(end - start))
	
	def process(self, file_path):
		with open(file_path, "r", encoding='utf-8') as f:
			file = json.load(f)
			self.doc_id += 1
			self.doc_id_to_url = file["url"]
			soup = BeautifulSoup(file["content"], "html.parser")
			token_to_count = self.tokenize(soup.get_text())
			for token, count in token_to_count.items():
				self.inverted_index[token][file["url"]] = count

			if debug:
				print("PROCESS - Document: {}, URL: {}".format(self.doc_id, file["url"]))

	def tokenize(self, text):
		token_to_count = defaultdict(int)
		tokens = word_tokenize(text)
		for t in tokens:
			token_to_count[self.ss.stem(t.lower())] += 1
		return token_to_count

	def dump(self):
		data = defaultdict(dict)
		with open(self.index_file, "w", encoding='utf-8') as f:
			json.dump(self.inverted_index, f)
		with open(self.lookup_table_file, "w", encoding='utf-8') as f:
			json.dump(self.inverted_index, f)



if __name__ == '__main__':
	dataset_path = "developer/DEV"
	index_file = "index.json"
	lookup_table_file = "lookup.json"
	debug = True

	i = Indexer(dataset_path, index_file, lookup_table_file, debug)
	i.start()