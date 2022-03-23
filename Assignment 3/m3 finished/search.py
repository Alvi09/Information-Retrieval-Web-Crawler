from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from math import log
import time
import json


class Search:
	def __init__(self, index_file, lookup_table_file, byte_offset_table, limit, debug):
		self.debug = debug

		self.log = open(index_file)

		with open(lookup_table_file, "r") as f:
			self.lookup_table = json.load(f)

		with open(byte_offset_table, "r") as f:
			self.byte_offset_table = json.load(f)

		self.ss = SnowballStemmer(language="english")
		self.limit = limit

	def poop(self, token, results):
		pee = "Displaying top {} results for \"{}\":".format(self.limit, token)
		for i in range(self.limit):
			pee += "\n\t{}. {}".format(i + 1, results[i])
		return pee


	def start(self):
		while True:
			query = str(input("Please enter a query string: "))

			if query == "!quit":
				break

			tokens = self.tokenize(query)

			if self.debug:
				start = time.perf_counter()

			documents = self.find_documents(tokens)

			if not documents:
				print("No documents found.")
			else:
				print(self.poop(query, documents))
			
			if self.debug:
				end = time.perf_counter()
				print("START - Time taken to retrieve query: {} ms.\n".format((end - start) * 1000))

		self.log.close()


	def tokenize(self, text):
		tokens = []
		for token in word_tokenize(text):
			if token.isalnum():
				tokens.append(self.ss.stem(token.lower()))

		if self.debug:
			print("TOKENIZE - Tokens: {}".format(tokens))

		return tokens

	def find_documents(self, tokens):
		# A dictionary with token keys to dictionary values in the form {id: freq}.
		token_doc_dict = {}

		# A list containing sets in the form {id}.
		id_list = []

		for t in tokens:
			self.log.seek(int(self.byte_offset_table[t]))
			line = self.log.readline()
			
			tmp = json.loads(line)

			id_list.append(set(tmp[t].keys()))
			token_doc_dict.update(tmp)

		# Nothing was found.
		if not token_doc_dict:
			return []

		id_list = sorted(id_list, key=lambda x: len(x))
		intersect = id_list[0].intersection(*id_list[1:])

		# res is a list of (doc_id, ft_idf_score) tuples.
		res = []

		for doc_id in intersect:
			total = 0
			for t in tokens:
				total += token_doc_dict[t][doc_id]
			res.append((doc_id, total))

		# Sort res by ft_idf_score from greatest to least and retrieve only the top results.
		res = sorted(res, key=lambda x: -x[1])[:self.limit]
		return list(map(lambda x: self.lookup_table[x[0]], res))



if __name__ == '__main__':
	index_file = "index.txt"
	lookup_table_file = "lookup.json"
	byte_offset_table = "byte_offset.json"

	limit = 5
	debug = True

	s = Search(index_file, lookup_table_file, byte_offset_table, limit, debug)
	s.start()
