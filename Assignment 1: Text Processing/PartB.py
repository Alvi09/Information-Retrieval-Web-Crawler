import sys
import PartA

def get_num_common_tokens(file1_lst, file2_lst):
    '''
    What's the time complexity?
        The way the '&' is used is that we have a smaller set that we want to check in the larger set if there are subsets of the smaller one in the larger one.
            This is O(min(file1_lst, file2_lst))
    '''
    return len(set(file1_lst) & set(file2_lst))

if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    imported_class = PartA.Word_Frequencies()

    my_file1_lst = imported_class.tokenize(open(file1))
    my_file2_lst = imported_class.tokenize(open(file2))

    print(get_num_common_tokens(my_file1_lst, my_file2_lst))
