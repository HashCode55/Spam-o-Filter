import re 
from collections import defaultdict
import math

def tokenize(message):
	message = message.lower()
	all_words = re.findall("[a-z0-9']+", message)
	#return a set of all words
	return set(all_words)

#print tokenize("Check whether this function is working or not")

def word_count(xtrain, ytrain):
	#initialise a dictionary containing a list for spam or nospam
	#0 is the list of spam and 1 is the list of nospam
	count_words = defaultdict(lambda : [0, 0])
	for ts, is_spam in zip(xtrain, ytrain):
		word_list = tokenize(ts)
		for words in word_list:
			count_words[words][0 if is_spam else 1] += 1
	return count_words
#the list will be like {"viagra : [5, 6]"}, through which we can interpret - viagra appears in 5 spam
#messages and 6 non-spam messages


#get the word probability in a list i.e. whether this word can be spam or nospam
def word_probability(xtrain, ytrain, total_spams, total_non_spams, k = 0.5):
	count_words = word_count(xtrain, ytrain)
	return [(w,
			(spam + k) / (2*k + total_spams), 
			(non_spams + k) / (2 * k  + total_non_spams))
	
			for w, (spam, non_spams) in count_words.items() ]

def spam_probability(word_probability, message):
	word_in_message = tokenize(message)
	log_prob_spam = log_prob_notspam = 0.0
	for word, spam_prob, notspam_prb in word_probability:
		#if the word is in the message update the counters accordingly
		if word in word_in_message:
			log_prob_spam += math.log(spam_prob)
			log_prob_notspam += math.log(notspam_prb)
		#if the word is not available in the message add the probability of not seeing it	
		else:
			log_prob_spam += math.log(1.0 - spam_prob) 	
			log_prob_notspam += math.log(1.0 - notspam_prb)

	#first we converted it into log so that the number remains small 
	#then we bring it back using exp		
	prob_spam = math.exp(log_prob_spam)		
	prob_notspam = math.exp(log_prob_notspam)

	return prob_spam / (prob_spam + prob_notspam)
