import Helper
class nb_classifier:
	#every class method takes a self paramete
	#k is the pseudocount for smoothing of the data
	def __init__(self, k = 0.5):
		self.k = k
		self.word_prob = []

	def fit(self, xtrain, ytrain):
		#count the number of spams and not spams 
		count_spam = len([spam for spam in ytrain if spam])
		count_notspam = len(ytrain) - count_spam

		#get the wordlist from the trained data 
		self.word_prob = Helper.word_probability(xtrain, ytrain, count_spam, count_notspam)	

	def predict(self, message):
		return Helper.spam_probability(self.word_prob, message)	


