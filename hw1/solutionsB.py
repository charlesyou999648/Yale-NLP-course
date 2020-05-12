import sys
import nltk
import math
import time
import collections
from collections import defaultdict
from collections import deque
from collections import Counter
import heapq
import itertools

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for sentence in brown_train:
        # tokens = sentence.strip().split()
        # sentence = START_SYMBOL + ' ' + START_SYMBOL + ' ' + sentence + STOP_SYMBOL
        tokens = sentence.strip().split()
        words, tags = [START_SYMBOL, START_SYMBOL], [START_SYMBOL, START_SYMBOL]
        for token in tokens:
            index = token.rfind('/')
            words.append(token[:index])
            tags.append(token[index+1:])
        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)

        brown_words.append(words)
        brown_tags.append(tags)
    # print(brown_words[0])
    return brown_words, brown_tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigram_tuples, trigram_tuples = [], []

    for tag in brown_tags:
        bigram_tuples.extend(nltk.bigrams(tag))
        trigram_tuples.extend(nltk.trigrams(tag))

    bigram_count = Counter(bigram_tuples)
    trigram_count = Counter(trigram_tuples)

    for trigram_tag in trigram_count:
        # print(bigram_count[trigram_tag[:2]], trigram_tag[:2])
        q_values[trigram_tag] = math.log2(trigram_count[trigram_tag] / float(bigram_count[trigram_tag[:2]]))

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    sorted(trigrams)
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    # print(brown_words)
    # print('brown_words',brown_words[0])
    # print(len(brown_words))
    word_list = []
    for word in brown_words:
        word_list.extend(word)

    word_counter = Counter(word_list)

    for key, value in word_counter.items():
        if value > RARE_WORD_MAX_FREQ:
            known_words.add(key)
    # print(known_words)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    for sentence in brown_words:
        new_sentence = []
        for token in sentence:
            # print(token)
            if token in known_words:
                new_sentence.append(token)
            else:
                new_sentence.append(RARE_SYMBOL)
        brown_words_rare.append(new_sentence)
        # print(brown_words_rare)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set (should not include start and stop tags)
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])

    word_tag_dict = {}
    tag_dict = {}

    for i in range(len(brown_tags)):
        sentence = brown_words_rare[i]
        tags = brown_tags[i]
        for j in range(len(tags)):
            word = sentence[j]
            tag = tags[j]
            if (word, tag) in word_tag_dict:
                word_tag_dict[(word, tag)] += 1
            else:
                word_tag_dict[(word, tag)] = 1
            if tag in tag_dict:
                tag_dict[tag] += 1
            else:
                tag_dict[tag] = 1

    for item in word_tag_dict:
        e_values[item] = math.log2(float(word_tag_dict[item])/tag_dict[item[1]])

    for tag in tag_dict:
        taglist.add(tag)

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    sorted(emissions)
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

def convert_tag(word, taglist, e_values):
    tags = []
    for tag in taglist:
        if (word, tag) in e_values:
            tags += [tag]
    return tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    """
    inspired by video on YouTube BY machine learning hub
    https://www.youtube.com/channel/UCB_JX4jH3QQmp69rmkWpl1A
    http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf
    """
    tagged = []
    # print(len(brown_dev_words))
    # print(brown_dev_words[0])

    for word_list in brown_dev_words:
        pi = collections.defaultdict() ## pi(k, u, v)
        bp = {}	## bp[(k, u, v)]: backpointers the argmax of pi[(k, u, v)]
        # Set pi(0, *, *) = 1
        pi[(0, START_SYMBOL, START_SYMBOL)] = 1.0
        update_word_list = []
        for word in word_list:
            if word in known_words:
                update_word_list += [word]
            elif word in word_list:
                update_word_list += [RARE_SYMBOL]
        n = len(update_word_list)

        for k in range(1, n + 1):
            m = k - 1
            v_word = update_word_list[m]
            v_tags = convert_tag(v_word, taglist, e_values)
            u_tags = [START_SYMBOL]
            if m > 0:
                u_word = update_word_list[m - 1]
                u_tags = convert_tag(u_word, taglist, e_values)
            w_tags = [START_SYMBOL]
            if m > 1:
                w_word = update_word_list[m - 2]
                w_tags = convert_tag(w_word, taglist, e_values)

            for u_tag in u_tags:
                for v_tag in v_tags:
                    best_prob = float('-Inf')
                    best_tag = None
                    for w_tag in w_tags:
                        if (v_word, v_tag) in e_values:
                            total_prob = ( pi.get((k - 1, w_tag, u_tag), LOG_PROB_OF_ZERO) + q_values.get((w_tag, u_tag, v_tag), LOG_PROB_OF_ZERO) + e_values.get((v_word, v_tag)) )
                            if total_prob > best_prob:
                                best_prob = total_prob
                                best_tag = w_tag
                    pi[(k, u_tag, v_tag)] = best_prob
                    bp[(k, u_tag, v_tag)] = best_tag

        best_prob = float('-Inf')
        u_word = update_word_list[n - 2]
        u_tags = convert_tag(u_word, taglist, e_values)
        v_word = update_word_list[n - 1]
        v_tags = convert_tag(v_word, taglist, e_values)

        for u_tag in u_tags:
            for v_tag in v_tags:
                total_prob = ( pi.get((n, u_tag, v_tag), LOG_PROB_OF_ZERO) + q_values.get((u_tag, v_tag, STOP_SYMBOL), LOG_PROB_OF_ZERO))
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_u_tag = u_tag
                    best_v_tag = v_tag

        tagged_words = []
        tagged_words.append(best_v_tag)
        tagged_words.append(best_u_tag)

        for i, k in enumerate(range(n - 2, 0, -1)):
            tagged_words.append(bp[(k + 2, tagged_words[i + 1], tagged_words[i])])
        tagged_words.reverse()

        full_words = []
        for i in range(0, n):
            full_words.append(word_list[i] + '/' + tagged_words[i])
        full_words.append('\n')
        tagged.append(' '.join(full_words))
        # print(tagged)
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in range(len(brown_words)) ]
    training = [list(x) for x in training]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff = default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)

    for words in brown_dev_words:
        tagged_words = []
        for word, tag in trigram_tagger.tag(words):
            tagged_words.append(word + '/' + tag)
        tagged_words.append('\n')
        tagged.append(' '.join(tagged_words))
    
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/' # absolute path to use the shared data
# DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print(f"Part B time: {str(time.clock())} sec")

if __name__ == "__main__": main()
