import glob, re
import os
from sklearn.cross_validation import train_test_split
import numpy as np
import Classifier

path = r"spam"

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

classifier = Classifier()
classifier.fit(xtrain)