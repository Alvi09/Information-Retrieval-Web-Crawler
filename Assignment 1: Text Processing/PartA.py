import sys
import re

class Word_Frequencies():
    def tokenize(self, TextFilePath):
        '''
        What's the time complexity?
            We loop over N lines in the TextFile
            We split() over M strings to parse
            Then, once we have those strings split in a list, we iterate over those M strings to append them

            TOTAL time complexity looks like O(N * M^2)
                
        Reads in text, and returns a List<Tokens>
            What is a token?
                A SEQUENCE of alphanumeric characters (i.e Apple, apple, aPpLe are the same token (can use REGEX for this))
            Do exception handling for bad inputs
        '''

        '''
        PLEASE NOTE!!! (TA'S / GRADERS)
            I DID NOT WRITE THIS REGEX TO PARSE THE STRING (i.e  re.split('[^a-zA-Z0-9]+')), I GOT IT FROM THIS!!
                https://www.techiedelight.com/split-string-with-delimiters-python/
        '''
        token_lst = []

        for line in TextFilePath:
            line = re.split('[^a-zA-Z0-9]+', line.lower())
            for string in line:
                if string.isalnum():
                    token_lst.append(string)        
        return token_lst

    def computeWordFrequencies(self, list_of_tokens):
        '''
        What's the time complexity?
            TOTAL is O(N) because we're only iterating over N strings in list_of_tokens, and lookups in dict is constant time

        Counts the number of occurrences of each token in a list and returns a Map<Token,Count>
            For example, {'apple', 100}
        '''

        freq_dict = {}

        for string in list_of_tokens:
            if string in freq_dict:
                freq_dict[string] += 1
            else:
                freq_dict[string] = 1
        
        return freq_dict
        
    def print(self, dict_of_freq):
        '''
        What's the time complexity?
            Python's sorting method is O(n log n)
                But we're iterating over the dictionary to print out the desired result output, so that's O(n)

            TOTAL is O(n log n + n) = O(N)

        Prints out the word frequency count in decreasing frequency
            For example, {'apple', 100, 'pear': 3}
        '''
        
        sorted_dict = dict(sorted(dict_of_freq.items(), key = lambda x: -x[1]))
        
        for token, freq in sorted_dict.items():
            print('{} -> {}'.format(token, freq))
            

if __name__ == '__main__':
    # Creating an instance of the class
    my_frequencies = Word_Frequencies()

    # Calling .tokenize()
    inputted_file = sys.argv[1]
    
    file = open(inputted_file, "r", encoding='utf-8')
    my_list_of_tokens = my_frequencies.tokenize(file)
    file.close()

    # Creating the dictionary
    my_word_freq = my_frequencies.computeWordFrequencies(my_list_of_tokens)

    # Sorting the dictionary
    my_print = my_frequencies.print(my_word_freq)

