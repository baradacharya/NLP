from gensim.models import Word2Vec
from nltk.corpus import brown
from sklearn.cluster import KMeans
import pickle

#brown.sents() #list of (list of str) #length 57340
#word_vec1 = Word2Vec(brown.sents())
#pickle.dump(word_vec1, open("word_vec", 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
word_vec = pickle.load(open("word_vec", 'rb'))
print "Words are converted to vector"
lis = []
words =set()

for i in xrange(100):
    if i % 1000 == 0:
        print i
    for word in brown.sents()[i]:
        if word.lower() not in words and word in word_vec:
            lis.append(word_vec[word])
            words.add(word.lower())

print "word vectors are ready for clustering"


kmeans = KMeans(n_clusters = 100, random_state=0).fit(lis)
# save the model to disk
filename = 'k_means.sav'
pickle.dump(kmeans, open(filename, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
word = 'is'
if word in word_vec.wv.vocab:
    print kmeans.predict(word_vec[word].reshape(1, -1))[0] #will flattern the array to one dimenssion
else:
    print "Doesn't find the word"
word = 'barada'
if word in word_vec.wv.vocab:
    print kmeans.predict(word_vec[word].reshape(1, -1))[0] #will flattern the array to one dimenssion
else:
    print "Doesn't find the word"

#loading model
loaded_model = pickle.load(open(filename, 'rb'))
print "from saved model"
print loaded_model.predict(word_vec['is'].reshape(1, -1))

