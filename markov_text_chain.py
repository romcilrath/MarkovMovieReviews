import re
import math
import pickle
import random
from numpy import int8, int16
from scipy.sparse import dok_matrix


# Loads the specified model
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.Unpickler(f).load()


# Takes a text chain (a list of words/ punctuation without spaces) and formats it for readable output
def print_chain(chain):
    # Regex to detect punctuation that needs no space before being written
    PUNCTUATION = re.compile('[.!?,;:]')

    # If the regex matches then don't pad the punctuation with a space
    # Otherwise it is a word and do pad the word with a space
    print(chain[0], end='')
    for word in chain[1:]:
        if PUNCTUATION.search(word):
            print(word, end='')
        else:
            print(' ' + word, end='')


# Helper method to print percentage complete to the console as the model is trained
def print_percentage(current, total):
    print(str(round(100 * current / total)) + " %")


# This constant is used to determine how many chunks to split the model into
# TODO: Just make this the size of the chunk
CHUNK_SIZER = 100000


class MarkovTextChainModel:

    # Initializer trains a model of length k from the raw file path
    def __init__(self, name, k, raw_file_path='data/movie_reviews.txt'):
        self.name = name
        self.k = k

        # Open the file to read them to a list
        with open(raw_file_path, 'r', encoding='utf-8') as file:
            file_lines = file.readlines()

        # Regular expressions to help remove unwanted punctuation
        REPLACE_NO_SPACE = re.compile('[\'\"()\[\]]')
        REPLACE_WITH_SPACE = re.compile('(<br\s*/><br\s*/>)|(-)|(/)')

        # Iterate over each line removing unwanted punctuation
        # Surround wanted punctuation with spaces so they will be treated as words later on
        # Split the line into a list of words/ punctuation and add them to the corpus
        corpus = list()
        for line in file_lines:
            line = REPLACE_NO_SPACE.sub("", line)
            line = REPLACE_WITH_SPACE.sub(" ", line)
            line = re.sub('([.!?,;:])', r' \1 ', line)
            corpus += line.split()

        # Get unique words and unique k words in the corpus as lists
        # Sort the unique k words to organize our chunks defined by how many unique k words we have
        self.unique_words = list(set(corpus))
        self.unique_k_words = set()
        for i in range(0, len(corpus) - self.k - 1):
            k_words = " ".join(corpus[i:i + self.k])
            self.unique_k_words.add(k_words)
        self.unique_k_words = list(sorted(self.unique_k_words))

        # By dividing the total number of unique k words by the chunk sizer and rounding up we get the chunk count
        # This is then used to determine chunk size, or how many elements are in each chunk
        self.chunk_count = math.ceil(len(self.unique_k_words) / CHUNK_SIZER)
        self.chunk_size = math.ceil(len(self.unique_k_words) / self.chunk_count)

        # Now that we have our unique words/ k words we need to be able to find their index in the list given the word
        self.unique_word_indices = dict()
        for i, word in enumerate(self.unique_words):
            self.unique_word_indices[word] = i
        self.unique_k_word_indices = dict()
        for i, k_words in enumerate(self.unique_k_words):
            self.unique_k_word_indices[k_words] = i

        # Define which k words act as boundaries for the chunks
        # This means later when looking up a set of k words we know which matrix in the matrix list
        self.chunk_boundaries = list()
        self.matrix_list = list()
        for i in range(0, len(self.unique_k_words), self.chunk_size):
            keys = self.unique_k_words[i:i + self.chunk_size]
            self.chunk_boundaries.append(keys[0])
            self.matrix_list.append(dok_matrix((self.chunk_size, len(self.unique_words)), dtype=int16))
        self.chunk_boundaries.append(self.unique_k_words[-1])

        # TODO: Remove
        print(self)

        for i in range(0, len(self.unique_k_words) - self.k - 1):
            if i % 20000 == 0:
                print_percentage(i, len(self.unique_k_words) - self.k - 1)
            k_words = " ".join(corpus[i:i + self.k])
            next_word = corpus[i + self.k]

            for j in range(0, len(self.chunk_boundaries) - 1):
                if self.chunk_boundaries[j] <= k_words < self.chunk_boundaries[j + 1]:
                    row = self.unique_k_word_indices[k_words] % self.chunk_size
                    col = self.unique_word_indices[next_word]
                    self.matrix_list[j][row, col] += 1

    # Output for printing the model
    def __str__(self):
        return "name:                 " + self.name + '\n' + \
               "k:                    " + str(self.k) + '\n' + \
               "unique k words count: " + str(len(self.unique_k_words)) + '\n' + \
               "unique word count:    " + str(len(self.unique_words)) + '\n' + \
               "chunk count:          " + str(self.chunk_count) + '\n' + \
               "chunk size:           " + str(self.chunk_size) + '\n' + \
               "chunk boundaries:     " + str(self.chunk_boundaries)

    def save_model(self, directory_path='models/'):
        file_path = directory_path + self.name + '.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def get_prob_word_pairs(self, k_words):
        for i in range(0, len(self.chunk_boundaries) - 1):
            if self.chunk_boundaries[i] <= k_words < self.chunk_boundaries[i + 1]:
                our_matrix = self.matrix_list[i]
                our_csr = our_matrix[self.unique_k_word_indices[k_words] % self.chunk_size, ]

                values = list(our_csr.values())
                keys = list()
                for key in our_csr.keys():
                    keys.append(self.unique_words[key[1]])

                return list(sorted(list(zip(values, keys)), reverse=True))

    # Given a string of k words this returns a next word based on the provided probabilities
    def fetch_next(self, k_words):
        prob_word_groups = self.get_prob_word_pairs(k_words)
        prob_sum = sum(abs(item[0]) for item in prob_word_groups)

        # Based on their occurrence values and a random roll assign the next word
        roll = random.uniform(0, prob_sum / 2)
        cum_sum = 0
        for i in range(0, len(prob_word_groups)):
            cum_sum += prob_word_groups[i][0]
            if roll <= cum_sum:
                return prob_word_groups[i][1]
            i += 1

    # Takes a string of k words as a seed and returns a chain of new words less than the given max length
    # If stop after full sentence is set to true then the chain will return after it detects an end sentence punctuation
    def stochastic_chain(self, seed, max_length=100, stop_after_full_sentence=True):

        # Regex to help detect the end of a sentence so we know not to add a space (no space before a period)
        END_SENTENCE = re.compile('[.!?]')

        # Initialize our chain with our seed and start by setting current to our seed
        chain = seed.split()
        current = seed

        # Until the max length for the chain is reached continue to fetch the next response word for the chain
        # Add that word to the chain then check if the end of a sentence has been reached then return the chain
        # Otherwise remove the first word from current and append the new response word so the process can be repeated
        for i in range(0, max_length):
            response = self.fetch_next(current)
            if response is None:
                chain.append('//ERROR//')
                return chain
            chain.append(response.replace(' ', ''))

            if stop_after_full_sentence and END_SENTENCE.search(response):
                return chain

            if self.k > 1:
                current = current.split(' ', 1)[1]
                current += " " + response
            else:
                current = response

            i += 1
        return chain
