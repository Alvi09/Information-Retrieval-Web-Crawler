from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from bs4 import BeautifulSoup
import os
import json
import time
import hashlib
import nltk
from math import log
nltk.download("punkt")



class Indexer:
    def __init__(self, dataset_path, index_file, lookup_table_file, byte_offset_table, debug):
        self.dataset_path = dataset_path
        self.index_file = index_file
        self.lookup_table_file = lookup_table_file
        self.byte_offset_table = byte_offset_table
        self.debug = debug

        self.ss = SnowballStemmer(language="english")

        self.doc_id = 1
        self.total = 0
        self.num_tokens = 0

        self.processed = 0

        self.lookup_table = dict()
        self.index =  defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.visited = set()

        self.alpha = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}


    def start(self):
        if debug: 
            start = time.perf_counter()
            print("START - Starting program...")

        for subdir, dirs, files in os.walk(self.dataset_path):
            for name in files:
                self.process(subdir + os.sep + name)

        self.final()

        if debug:
            end = time.perf_counter()
            print("START - Indexed {} out of {} documents ".format(self.doc_id, self.total))
            print("START - Total number of tokens: {}".format(self.num_tokens))
            print("START - Program execution time: {} minutes".format((end - start) / 60))
    
    def process(self, file_path):
        with open(file_path, "r") as f:
            file = json.load(f)
            url = file["url"]

            if url.find("#") != -1:
                url = url[:url.find("#")]

            if url not in self.visited:
                self.lookup_table[self.doc_id] = url

                soup = BeautifulSoup(file["content"], "html.parser")

                for elem in soup(['style', 'script']):
                    elem.extract()

                token_freq = self.get_token_freq(soup.get_text())

                seen = set()
                weighted_tags = [["a", "b", "strong", "h2", "h3", "h4", "h5", "h6"], ["title", "h1"]]

                for group in range(len(weighted_tags)):
                    for tag in weighted_tags[group]:
                        res = soup.find(tag)
                        
                        if not res:
                            continue

                        token_list = self.tokenize(soup.find(tag).get_text())

                        for token in token_list:
                            if token not in seen:
                                seen.add(token)
                                if token[0] in self.alpha:
                                    token_freq[token[0]][token] += ((group + 1) * 2.5)
                                    token_freq[token[0]][token] *= ((group + 1) * 1.5)
                                else: 
                                    token_freq["+"][token] += ((group + 1) * 2.5)
                                    token_freq["+"][token] *= ((group + 1) * 1.5)

                for token_letter, token_freq_dict in token_freq.items():
                    for token, freq in token_freq_dict.items():
                        self.index[token_letter][token][self.doc_id] = freq

                if debug:
                    print("PROCESS - Document: {}, URL: {}".format(self.doc_id, file["url"]))

                self.doc_id += 1
                self.visited.add(url)

                self.total += 1
                self.processed += 1

                if self.processed > 15000:
                    self.dump()

    def get_token_freq(self, text):
        token_freq = defaultdict(lambda: defaultdict(int))
        token_list = word_tokenize(text)
        for token in token_list:
            if token.isalnum():
                if token[0].lower() in self.alpha:
                    token_freq[token[0].lower()][self.ss.stem(token.lower())] += 1
                else:
                    token_freq["+"][self.ss.stem(token.lower())] += 1
        return token_freq

    def tokenize(self, text):
        token_list = word_tokenize(text)
        token_list = list(map(lambda token: self.ss.stem(token.lower()), token_list))
        return filter(lambda token: token.isalnum(), token_list)


    def dump(self):
        if debug:
            print("DUMP - Writing information to files.")

        for char, token_freq in self.index.items():
            filename = "partial_index_" + char + ".json"
            try:
                with open(filename, "r+") as file:
                    j = json.load(file)

                    for t, freq in j.items():
                        if t in token_freq:
                            token_freq[t].update(freq)
                        else:
                            token_freq[t] = freq

                with open(filename, "w+") as f:
                    if debug:
                        print("DUMP - Merging index with {}".format(filename))
                    json.dump(token_freq, f)

            except FileNotFoundError:
                filename = "partial_index_" + char + ".json"
                if debug:
                    print("DUMP - Creating file: {}".format(filename))
                with open(filename, "w+") as f:
                    json.dump(token_freq, f)
            

        self.processed = 0
        self.index =  defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def calculate_tf_idf(self, freq, length):
        tf = 1 + log( freq )
        idf = log( self.doc_id / length ) 
        return tf * idf

    def final(self):
        with open(self.lookup_table_file, "w+") as file:
            if debug:
                print("FINAL - Dumping lookup_table to {}".format(self.lookup_table_file))

            json.dump(self.lookup_table, file)

        self.dump()

        offset_dict = {}

        if debug:
            print("FINAL - Merging indexes...")

        with open(self.index_file, "a+") as f_idx:

            curr_offset = f_idx.tell()


            for char in 'abcdefghijklmnopqrstuvwxyz+':
                filename = "partial_index_" + char + ".json"

                with open(filename, "r") as f:
                    bucket = json.load(f)
                    for token, did_freq in bucket.items():
                        current = defaultdict(lambda: defaultdict(int))

                        for did, freq in did_freq.items():
                            if not freq:
                                current[token][did] = self.calculate_tf_idf(1, len(did_freq))
                            else:
                                current[token][did] = self.calculate_tf_idf(freq, len(did_freq))
                                
                        self.num_tokens += 1

                        f_idx.write(json.dumps(current))
                        f_idx.write("\n")

                        offset_dict[token] = curr_offset
                        curr_offset = f_idx.tell()

                os.remove(filename)

        
        with open(self.byte_offset_table, "w+") as file:
            if debug:
                print("FINAL - Dumping byte_offset_table to {}".format(self.byte_offset_table))
            json.dump(offset_dict, file)


if __name__ == '__main__':
    dataset_path = "DEV"
    index_file = "index.txt"
    lookup_table_file = "lookup.json"
    byte_offset_table = "byte_offset.json"
    debug = True

    i = Indexer(dataset_path, index_file, lookup_table_file, byte_offset_table, debug)
    i.start()
