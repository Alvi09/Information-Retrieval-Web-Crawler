from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from bs4 import BeautifulSoup
import os
import json
import time
import nltk
from math import log
import re
from simhash import Simhash, SimhashIndex
nltk.download("punkt")




class Indexer:
    def __init__(self, dataset_path, index_file, lookup_table_file, byte_offset_table, debug):
        self.dataset_path = dataset_path
        self.index_file = index_file
        self.lookup_table_file = lookup_table_file
        self.byte_offset_table = byte_offset_table
        self.debug = debug

        # This is the Porter2 stemmer.
        self.ss = SnowballStemmer(language="english")

        # Total number of unique documents found.
        self.doc_id = 1

        # Total number of documents found.
        self.total = 0

        # Total number of tokens found.
        self.num_tokens = 0

        # Current number of processed documents before off loading to memory.
        self.processed = 0

        # { doc_id::int -> url::str }
        self.lookup_table = dict()

        # { starting_char::str -> { token::str -> { doc_id:: int -> freq::int } } }
        self.index =  defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Set containing document URLs to prevent revisting the same document.
        self.visited = set()

        self.alpha = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}

        self.hashes = None

        self.duplicates = []


    def start(self):
 
        if debug: 
            start = time.perf_counter()
            print("START - Starting program...")

        # Go through each folder in the provided dataset folder.
        for subdir, dirs, files in os.walk(self.dataset_path):
            # Go through each file in the folder.
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
            self.total += 1

            file = json.load(f)
            url = file["url"]

            # Defrag the URL to prevent visting the same document twice.
            if url.find("#") != -1:
                url = url[:url.find("#")]

            # Check whether the URL has been visited.
            if url not in self.visited:
                self.lookup_table[self.doc_id] = url

                # Parse the document using BS4.
                soup = BeautifulSoup(file["content"], "html.parser")

                # Remove low content tags.
                for elem in soup(['style', 'script']):
                    elem.extract()

                # Creating simhash of pages to check for near/exact duplicates
                index = Simhash(self.get_features(soup.get_text()))
                ifDuplicate = True
                if not self.hashes:
                    self.hashes = SimhashIndex([(str(self.total), index)], k=2)
                else:
                    ifDuplicate = self.hashes.get_near_dups(index)
                    self.hashes.add(str(self.total), index)
                if not ifDuplicate or self.total == 1:

                    # { starting_char::str -> { token::str -> freq::int } }
                    token_freq = self.get_token_freq(soup.get_text())

                    # Set containing tokens to prevent changing the weight group of a token on the second encounter.
                    seen = set()

                    # List of tags categorized by importance.
                    weighted_tags = [["title", "h1"], ["a", "b", "strong", "h2", "h3", "h4", "h5", "h6"]]

                    # { group::int -> ( flat::int, multiplier::int ) }
                    weights = {0: (5, 3), 1: (2.5, 1.5)}

                    # Go through each tag and parse for tokens.
                    for group in range(len(weighted_tags)):
                        for tag in weighted_tags[group]:
                            res = soup.find(tag)

                            # The tag was not found and can be skipped.
                            if not res:
                                continue

                            # List containing all tokens found for a specific tag.
                            token_list = self.tokenize(soup.find(tag).get_text())

                            for token in token_list:
                                if token not in seen:
                                    bucket = token[0]
                                    f, m = weights[group]

                                    if bucket in self.alpha:
                                        # Add a base weight to the frequency.
                                        token_freq[bucket][token] += f
                                        # Add a multiplier weight to the frequency.
                                        token_freq[bucket][token] *= m
                                    else:
                                        token_freq["+"][token] += f
                                        token_freq["+"][token] *= m

                                    seen.add(token)

                    # Combine the current document's inverted index to the main inverted index.
                    for token_letter, token_freq_dict in token_freq.items():
                        for token, freq in token_freq_dict.items():
                            self.index[token_letter][token][self.doc_id] = freq

                    if debug:
                        print("PROCESS - Document: {}, URL: {}".format(self.doc_id, file["url"]))

                    self.doc_id += 1
                    self.visited.add(url)
                    self.processed += 1

                    # Threshhold value representing when to offload the index to partial indices on disk.
                    if self.processed > 15000:
                        self.dump()

    def get_token_freq(self, text):
        # { starting_char::str -> { token::str -> freq::int } }
        token_freq = defaultdict(lambda: defaultdict(int))
        # This is a tokenize function from nltk library that returns a list of tokens.
        token_list = word_tokenize(text)
        for token in token_list:
            if token.isalnum():
                # Separate tokens into different buckets depending on starting character.
                if token[0].lower() in self.alpha:
                    # Convert the token to lowercase and stem it.
                    token_freq[token[0].lower()][self.ss.stem(token.lower())] += 1
                else:
                    # All other characters will go into a separate bucket.
                    token_freq["+"][self.ss.stem(token.lower())] += 1
        return token_freq

    def tokenize(self, text):
        # This is a tokenize function from nltk library that returns a list of tokens.
        token_list = word_tokenize(text)
        # Convert each token to it's lower case stem.
        token_list = list(map(lambda token: self.ss.stem(token.lower()), token_list))
        # Keep only alphanumeric tokens.
        return filter(lambda token: token.isalnum(), token_list)


    def dump(self):
        if debug:
            print("DUMP - Writing information to files.")

        for char, token_freq in self.index.items():
            # The name of the partial index file on disk to be offloaded to.
            filename = "partial_index_" + char + ".json"
            try:
                with open(filename, "r+") as file:
                    # Load the partial index file from disk and combine it with the one in memory.
                    j = json.load(file)
                    for t, freq in j.items():
                        if t in token_freq:
                            token_freq[t].update(freq)
                        else:
                            token_freq[t] = freq

                # Create the partial index file and dump the combined partial index.
                with open(filename, "w+") as f:
                    if debug:
                        print("DUMP - Merging index with {}".format(filename))
                    json.dump(token_freq, f)

            except FileNotFoundError:
                filename = "partial_index_" + char + ".json"
                if debug:
                    print("DUMP - Creating file: {}".format(filename))
                # Create the file and dump the partial index.
                with open(filename, "w+") as f:
                    json.dump(token_freq, f)
            

        self.processed = 0
        self.index =  defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def calculate_tf_idf(self, freq, length):
        # 1 + log( token frequency )
        tf = 1 + log( freq )
        # log( total number of documents / token document frequency )
        idf = log( self.doc_id / length ) 
        return tf * idf

    def final(self):
        # Dump the lookup table in memory to a file on disk.
        with open(self.lookup_table_file, "w+") as file:
            if debug:
                print("FINAL - Dumping lookup_table to {}".format(self.lookup_table_file))

            json.dump(self.lookup_table, file)

        # Dump any remaining index in memory to it's respective partial index file.
        self.dump()

        # { token::str -> byte_offset::int }
        offset_dict = {}

        if debug:
            print("FINAL - Merging indexes...")

        print(self.duplicates)

        # Create the final index file for appending, does not load the entire file into memory.
        with open(self.index_file, "a+") as f_idx:
            # Get the current byte offset.
            curr_offset = f_idx.tell()

            # Go through each partial index bucket.
            for char in 'abcdefghijklmnopqrstuvwxyz+':
                filename = "partial_index_" + char + ".json"

                with open(filename, "r") as f:
                    # Load the partial index into memory.
                    bucket = json.load(f)


                    # The partial indexes stored on disk are in the form { token::str -> { doc_id:: int -> freq::int } } }.
                    # Given the total number of documents has now been recognized, update the frequency to tf-idf score.
                    for token, did_freq in bucket.items():
                        # { token::str -> { doc_id:: int -> tf-idf score::int } } }
                        current = defaultdict(lambda: defaultdict(int))
                        # { doc_id:: int -> freq::int }
                        for did, freq in did_freq.items():
                            # Convert each frequency into a tf-idf score and store it.
                            if not freq:
                                current[token][did] = self.calculate_tf_idf(1, len(did_freq))
                            else:
                                current[token][did] = self.calculate_tf_idf(freq, len(did_freq))
                        
                        self.num_tokens += 1

                        # Write the updated partial index in memory to the index file on disk.
                        f_idx.write(json.dumps(current))
                        f_idx.write("\n")

                        # Record the byte offset pointing to the current token.
                        offset_dict[token] = curr_offset
                        # Update the byte offset.
                        curr_offset = f_idx.tell()

                # Remove the partial index from disk.
                os.remove(filename)

        # Dump the byte offset table in memory to a file on disk.
        with open(self.byte_offset_table, "w+") as file:
            if debug:
                print("FINAL - Dumping byte_offset_table to {}".format(self.byte_offset_table))
            json.dump(offset_dict, file)

    def get_features(self, s):
        width = 3
        s = s.lower()
        s = re.sub(r'[^\w]+', '', s)
        return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]


if __name__ == '__main__':
    # The name of the folder containing the provided dataset.
    dataset_path = "DEV"

    # Desired names of files to be created containing the respective data structure name. 
    index_file = "index.txt"
    lookup_table_file = "lookup.json"
    byte_offset_table = "byte_offset.json"

    debug = True

    i = Indexer(dataset_path, index_file, lookup_table_file, byte_offset_table, debug)
    i.start()
