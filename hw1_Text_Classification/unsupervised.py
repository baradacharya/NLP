import tarfile
import matplotlib.pyplot as plt

def read_instance(tar, ifname):
	inst = tar.getmember(ifname)
	ifile = tar.extractfile(inst)
	content = ifile.read().strip()
	return content

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
	plt.title("Unsupervised classifier")
	fig.tight_layout()
	plt.show()
	plt.savefig("confusion_unsupervised.png")

##first processing the data
tarfname = "data/speech.tar.gz"
tar = tarfile.open(tarfname, "r:gz") ##include tarfname
class Data: pass
speech = Data()

print("-- train data")
speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(tar, "train.tsv")
print(len(speech.train_data))
print("-- dev data")
speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(tar, "dev.tsv")
print(len(speech.dev_data))
print("-- transforming data and labels")

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
speech.count_vect = TfidfVectorizer(ngram_range=(1,3),stop_words='english')
speech.trainX = speech.count_vect.fit_transform(speech.train_data)
speech.devX = speech.count_vect.transform(speech.dev_data)
from sklearn import preprocessing
speech.le = preprocessing.LabelEncoder()
speech.le.fit(speech.train_labels)
speech.target_labels = speech.le.classes_
speech.trainy = speech.le.transform(speech.train_labels)
speech.devy = speech.le.transform(speech.dev_labels)

#unlabled data

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


##training the data

import numpy as np
import scipy.sparse as sc

trainX = speech.trainX
trainY = speech.trainy

dec = True
itr = 0
prev = 0

unl = []
for i in range(unlabeled.X.shape[0]):
	temp = unlabeled.X[i]
	unl.append(temp)

unlX = []
# Try out different cutoff values
# cutoof_values = [0.5, 0.75, 0.9, 0.95]
# dev_accs =[]
# unlabel_transforms = []
#
# for cutoff in cutoof_values:
# 	print("cut-off value",cutoff)
dev_acc = []
unlabel_transform = []
itr = 1
while dec:
	print("itr", itr)
	itr += 1;
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(penalty='l2', multi_class='multinomial', class_weight='balanced', solver='lbfgs', C = 8.1)
	cls.fit(trainX, trainY)
	from sklearn import metrics
	y_hat = cls.predict(trainX)
	acc = metrics.accuracy_score(trainY,y_hat)
	print("Training Accuracy", acc)
	# print("shape of trainX and trainY before update",trainX.shape,trainY.shape)

	yp = cls.predict(speech.devX)
	acc = metrics.accuracy_score(speech.devy, yp)
	print("Validation Accuracy",acc)
	dev_acc.append(acc)

	if(acc > 0.5 or itr > 3 or abs(acc - prev) < 0.0001):
	#if (itr > 10):
		# print(acc - prev)
		print("Loop Stops.")
		dec = False

	for i in range(len(unl)):
		mat = cls.predict_proba(unl[i])
		flag = True
		for j in range(mat.shape[1]):
			if(mat[0][j] >= 0.7):
				trainX = sc.vstack((trainX, unl[i]))
				y_hat = cls.predict(unl[i])
				trainY = np.hstack((trainY, y_hat[0]))
				flag = False
				break

		#element have to train once more
		if flag== True:
			unlX.append(unl[i])

	# print("length of unlabeled data before labeling",len(unl))
	transform = len(unl) - len(unlX)
	unlabel_transform.append(transform)
	unl = unlX
	unlX =[]
	prev = acc

	# print("length of unlabeled data after labeling", len(unl))
	# print("shape of trainX and trainY before update", trainX.shape, trainY.shape)
# dev_accs.append(dev_acc)
print(dev_acc)
# unlabel_transforms.append(unlabel_transform)
print(unlabel_transform)
##predcting the output and writing to file
cls = LogisticRegression(penalty='l2', multi_class='ovr', class_weight='balanced', solver='lbfgs', C = 8.1)
cls.fit(trainX, trainY)
# y_hat = cls.predict(trainX)
# acc = metrics.accuracy_score(trainY,y_hat)
# print("Training Accuracy", acc)
# print("shape of trainX and trainY before update",trainX.shape,trainY.shape)

yp = cls.predict(speech.devX)
acc = metrics.accuracy_score(speech.devy, yp)
print("Validation Accuracy",acc)
m = metrics.confusion_matrix(speech.devy,yp)

plot_confusion_matrix(m,speech.le.classes_)

#write_pred_kaggle_file(unlabeled, cls, "data/speech-pred.csv", speech)

