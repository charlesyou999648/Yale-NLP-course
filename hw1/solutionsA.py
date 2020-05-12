from collections import deque
from collections import defaultdict
import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):

	unigram_p = {}
	bigram_p = {}
	trigram_p = {}

	unigrams = {}
	bigrams = {}
	trigrams = {}
    
	uni_count = 0
	tc_count = 0
	training_corpus_list = []

	for sentence in training_corpus:
		sentence = START_SYMBOL + " " + START_SYMBOL + " " + sentence + " " + STOP_SYMBOL
		training_corpus_list.append(sentence)
		tc_count += 1

	for sentence in training_corpus_list:
		tokens =sentence.strip().split()
		# print(tokens) # [*, *, a, b, c]
		for i in range(2,len(tokens)):
			unigram = (tokens[i],)
			if unigram != START_SYMBOL:
				uni_count += 1
				if unigram in unigram_p:
					unigram_p[unigram] += 1
					unigrams[unigram] +=1 
				else:
					unigram_p[unigram] = 1
					unigrams[unigram] = 1

			bigram = (tokens[i-1],tokens[i])
			if bigram != (START_SYMBOL, START_SYMBOL):
				if bigram in bigram_p:
					bigram_p[bigram] += 1
					bigrams[bigram] += 1
				else:
					bigram_p[bigram] = 1
					bigrams[bigram] = 1
				
				trigram = (tokens[i-2],tokens[i-1],tokens[i])
				if trigram in trigram_p:
					trigram_p[trigram] += 1
					trigrams[trigram] += 1
				else:
					trigram_p[trigram] = 1
					trigrams[trigram] = 1

	# unigram
	for keys in set(unigram_p):
		unigram_p[keys] = math.log((unigram_p[keys]),2) - math.log(uni_count,2)
	
	# bigram
	for keys in bigrams:
		if keys[0] == START_SYMBOL:
			bigram_p[keys] = math.log(bigrams[keys],2) - math.log(tc_count,2)
		else:
			bigram_p[keys] = math.log(bigrams[keys],2) - math.log(unigrams[(keys[0]),],2)
	
	# trigram
	for keys in trigrams:
		if keys[0] == START_SYMBOL and keys[1] == START_SYMBOL:
			trigram_p[keys] = math.log(trigrams[keys],2) - math.log(tc_count,2)
		else:
			trigram_p[keys] = math.log(trigrams[keys],2) - math.log(bigrams[(keys[0],keys[1])],2)

	# ('far-out',): -19.410265006819234
	# print(unigram_p)
	return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    #output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    sorted(unigrams_keys)
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    sorted(bigrams_keys)
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    sorted(trigrams_keys)
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
	scores = []

	for sentence in corpus:
		sum_prob_uni = 0
		sum_prob_bi = 0
		sum_prob_tri = 0

		# unigram
		if n == 1:
			# print (sentence)
			sentence += STOP_SYMBOL
			# print ('******' + sentence)
			tokens = sentence.strip().split()
			# print(tokens)
			for word in tokens:
				if tuple([word]) in ngram_p:
					sum_prob_uni += ngram_p[(word,)]
				else:
					sum_prob_uni = MINUS_INFINITY_SENTENCE_LOG_PROB
					break
			scores.append(sum_prob_uni)
    	
    	# bigram
		if n == 2:
			sentence = START_SYMBOL + ' ' + sentence + STOP_SYMBOL
			tokens = sentence.strip().split()
			# print('bigram',tokens)
			for w1,w2 in zip(tokens[0::1], tokens[1::1]):
                #print ngram_p[(w1, w2)]
				if (w1, w2) in ngram_p:
					sum_prob_bi += ngram_p[(w1, w2)]
				else:
					sum_prob_bi = MINUS_INFINITY_SENTENCE_LOG_PROB
					break
			scores.append(sum_prob_bi)
        
        # trigram
		if n == 3:
			sentence = START_SYMBOL + ' ' + START_SYMBOL + ' ' + sentence + STOP_SYMBOL
			tokens = sentence.strip().split()
			for w1, w2, w3 in zip(tokens[0::1], tokens[1::1], tokens[2::1]):
				if (w1, w2, w3) in ngram_p:
					sum_prob_tri += ngram_p[(w1, w2, w3)]
				else:
					sum_prob_tri = MINUS_INFINITY_SENTENCE_LOG_PROB
					break
			scores.append(sum_prob_tri)
	return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
	scores = []
	for sentence in corpus:
		sentence_score = 0
		sentence = START_SYMBOL + ' ' + START_SYMBOL + ' ' + sentence + STOP_SYMBOL
		tokens = sentence.strip().split()

		for w1, w2, w3 in zip(tokens[0::1], tokens[1::1], tokens[2::1]):
			score_uni = 0
			score_bi = 0
			score_tri = 0

			# unigram
			if tuple([w3]) in unigrams:
				score_uni = 2**unigrams[tuple([w3])]
			# bigram
			if (w2, w3) in bigrams:
				score_bi = 2**bigrams[(w2, w3)]
			# trigrams
			if (w1, w2, w3) in trigrams:
				score_tri = 2**trigrams[(w1, w2, w3)]

			if (score_uni != 0) or (score_bi != 0) or (score_tri != 0):
				word_score = math.log((score_uni + score_bi + score_tri), 2) + math.log(1, 2) - math.log(3, 2)
				sentence_score += word_score
			else:
				sentence_score = MINUS_INFINITY_SENTENCE_LOG_PROB
				break
		scores.append(sentence_score)
	return scores

DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
# DATA_PATH = 'data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print(f"Part A time: {str(time.clock())} sec")

if __name__ == "__main__": main()
