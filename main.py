import glob, re
import os
from sklearn.cross_validation import train_test_split
import numpy as np
from Classifier import nb_classifier
from collections import Counter

path = r"spam/*"

data = []
spam_nospam = []
#is_spam = []
#count = 0
for files in glob.glob(os.path.join(path, '*.*')):
	is_spam = "ham" not in files
	with open(files, 'r', encoding='utf-8', errors='ignore') as file:
		for line in file:
			if line.startswith("Subject:"):
				#remove the leading stuff after subject, re.sub is an important function	
				subject = re.sub(r"^Subject: ", "", line).strip()
				data.append(subject)
				spam_nospam.append(is_spam)

data = np.asarray(data)
spam_nospam = np.asarray(spam_nospam)
#print count

#NOW this data is the data set we have
#we split it using cross validation into training set and testing set

xtrain, xtest, ytrain, ytest = train_test_split(data, spam_nospam, test_size = 0.25)
classifier = nb_classifier()
classifier.fit(xtrain, ytrain)

predictions = [(message, is_spam, classifier.predict(message))
			for message, is_spam in zip(xtest, ytest)]
 #we assume that if the perdiction returns > 0.5 it'll be a spam
details = Counter(((is_spam, spam_probability > 0.5)
 			for _, is_spam, spam_probability in predictions))
#this is our accuracy but we are more interested in precision and accuracy 
print ("Accuracy of the test is:", ((details[(False, False)] + details[(True, True)]) / ytest.size) * 100)
#precision is how accurate were the positive prediction
print ("Precision of the test is:", (details[(True, True)] / (details[(True, True)] + details[(False, True)])) * 100)
#recall is what is the fraction of positive predictions made
print ("Recall of the test is:", (details[(True, True)] / (details[(True, True)] + details[(True, False)])) * 100)

