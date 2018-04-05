#!/bin/python
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nltk.stem.snowball import EnglishStemmer

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

def plot(m):
	plt.figure()
	img=plt.imshow(m)
    #img.set_clim(0.0,1.0)
	img.set_interpolation('nearest')
    #plt.set_cmap('gray')
	plt.colorbar()

def plot_confusion_matrix(m,cls):
	fig = plt.figure()
	img = plt.imshow(m)
	tick_marks = np.arange(len(cls))
	plt.xticks(tick_marks, cls,rotation = 90)
	plt.yticks(tick_marks, cls)
	# img.set_clim(0.0,1.0)
	img.set_interpolation('nearest')
	# plt.set_cmap('gray')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.title("Supervised classifier")
	fig.tight_layout()
	plt.show()
	plt.savefig("confusion_supervised.png")


def pca(m, k):
    from numpy.linalg import svd
    from numpy.linalg import eig
    from numpy.linalg import det
    u,s,v = svd(m)
    rs = np.sqrt(np.diag(s[:k]))
    x=np.dot(u[:,:k], rs)
    y=np.dot(rs, v[:k])
    mhat=np.dot(x, y)
    return s, x, y, mhat




def read_files(tarfname):
	"""Read the training and development data from the speech tar file.
	The returned object contains various fields that store the data, such as:

	train_data,dev_data: array of documents (array of words)
	train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
	train_labels,dev_labels: the true string label for each document (same length as data)

	The data is also preprocessed for use with scikit-learn, as:

	count_vec: CountVectorizer used to process the data (for reapplication on new data)
	trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
	le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
	target_labels: List of labels (same order as used in le)
	trainy,devy: array of int labels, one for each document
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	speech = Data()
	print("-- train data")
	speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(tar, "train.tsv")
	print(len(speech.train_data))
	print("-- dev data")
	speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(tar, "dev.tsv")
	print(len(speech.dev_data))
	print("-- transforming data and labels")

	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfVectorizer
	#speech.count_vect = TfidfVectorizer()
	speech.count_vect = CountVectorizer(ngram_range=(1,2))

	#stemmer = EnglishStemmer()
	#stemmer = PorterStemmer()
	stemmer = nltk.stem.SnowballStemmer('english')
	analyzer = CountVectorizer().build_analyzer()

	def stemmed_words(doc):
		return (stemmer.stem(w) for w in analyzer(doc))


	#Output = [stemmer.stem(word) for word in tokens]

	#speech.count_vect = CountVectorizer(ngram_range=(1,2), analyzer=stemmed_words)

	#speech.count_vect = TfidfVectorizer(ngram_range=(1,3),analyzer=stemmed_words,sublinear_tf = True,smooth_idf = True)
	#speech.count_vect = TfidfVectorizer(ngram_range=(1, 2),analyzer=stemmed_words)
	speech.trainX = speech.count_vect.fit_transform(speech.train_data)
	speech.devX = speech.count_vect.transform(speech.dev_data)


	from sklearn import preprocessing
	speech.le = preprocessing.LabelEncoder()
	speech.le.fit(speech.train_labels)
	k = len(speech.le.classes_)

	speech.target_labels = speech.le.classes_
	speech.trainy = speech.le.transform(speech.train_labels)
	speech.devy = speech.le.transform(speech.dev_labels)

	# D = speech.devX.shape[0]
	# N = speech.devX.shape[1]
	# m = np.zeros((N, D))
    #
	# for i in range(D):
	# 	d = speech.devX[i]
	# 	for w,score in d:
	# 		m[w][i] = score
	# return m
    #
	# s, wv, dv, mhat = pca(m, k)
	# plot(dv)
	# plt.savefig("lsa-docv.png")

	tar.close()
	return speech


def read_unlabeled(tarfname, speech):
	"""Reads the unlabeled data.

	The returned object contains three fields that represent the unlabeled data.

	data: documents, represented as sequence of words
	fnames: list of filenames, one for each document
	X: bag of word vector for each document, using the speech.vectorizer
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []
	unlabeled.fnames = []
	for m in tar.getmembers():
		if "unlabeled" in m.name and ".txt" in m.name:
			unlabeled.fnames.append(m.name)
			unlabeled.data.append(read_instance(tar, m.name))
	unlabeled.X = speech.count_vect.transform(unlabeled.data)
	print(unlabeled.X.shape)
	tar.close()
	return unlabeled

def read_tsv(tar, fname):
	member = tar.getmember(fname)
	print(member.name)
	tf = tar.extractfile(member)
	data = []
	labels = []
	fnames = []
	for line in tf:
		line = line.decode("utf-8")
		(ifname,label) = line.strip().split("\t")
		#print ifname, ":", label
		content = read_instance(tar, ifname)
		labels.append(label)
		fnames.append(ifname)
		data.append(content)
	return data, fnames, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
	"""Writes the predictions in Kaggle format.

	Given the unlabeled object, classifier, outputfilename, and the speech object,
	this function write the predictions of the classifier on the unlabeled data and
	writes it to the outputfilename. The speech object is required to ensure
	consistent label names.
	"""
	yp = cls.predict(unlabeled.X)
	labels = speech.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	for i in range(len(unlabeled.fnames)):
		fname = unlabeled.fnames[i]
		# iid = file_to_id(fname)
		f.write(str(i+1))
		f.write(",")
		#f.write(fname)
		#f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()

def file_to_id(fname):
	return str(int(fname.replace("unlabeled/","").replace("labeled/","").replace(".txt","")))

def write_gold_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the truth.

	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			# iid = file_to_id(ifname)
			i += 1
			f.write(str(i))
			f.write(",")
			#f.write(ifname)
			#f.write(",")
			f.write(label)
			f.write("\n")
	f.close()

def write_basic_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the naive baseline.

	This baseline predicts OBAMA_PRIMARY2008 for all the instances.
	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write("OBAMA_PRIMARY2008")
			f.write("\n")
	f.close()

def read_instance(tar, ifname):
	inst = tar.getmember(ifname)
	ifile = tar.extractfile(inst)
	content = ifile.read().strip()
	return content

if __name__ == "__main__":
	print("Reading data")
	tarfname = "data/speech.tar.gz"
	speech = read_files(tarfname)
	print("Training classifier")
	import classify
	# #C = [100,50,20,10,9,8,7,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
	# C = [1000,500,300,200,150,120,110,105,100, 50, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
	# i_c = [1/i for i in C]
	train_accs = []
	test_accs = []
	c =1
	# for c in C:
	cls = classify.train_classifier(speech.trainX, speech.trainy,c)
	confusion_mtrx,train_acc = classify.evaluate(speech.trainX, speech.trainy, cls)
	train_accs.append(train_acc)
	confusion_mtrx,test_acc = classify.evaluate(speech.devX, speech.devy, cls)
	test_accs.append(test_acc)
	#plot_confusion_matrix(confusion_mtrx,speech.le.classes_)

	#get important features for class
	for i in range(0, cls.coef_.shape[0]):
		top10_indices = np.argsort(cls.coef_[i])[-10:]
		top10_feature = []
		print(speech.le.classes_[i])
		for idx in top10_indices:
			for word in speech.count_vect.vocabulary_:
				if(speech.count_vect.vocabulary_[word] == idx):
					top10_feature.append(word)
		print(top10_feature)

	#print(i_c)
	print(train_accs)
	print(test_accs)
	print("Reading unlabeled data")
	unlabeled = read_unlabeled(tarfname, speech)
	print("Writing pred file")
	write_pred_kaggle_file(unlabeled, cls, "data/speech-pred.csv", speech)

	# You can't run this since you do not have the true labels
	# print "Writing gold file"
	# write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
	# write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")
