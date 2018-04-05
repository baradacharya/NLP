#!/bin/python

import pickle
from gensim.models import Word2Vec
from nltk.corpus import brown
class Cluster:
	def __init__(self):
		self.model = pickle.load(open('k_means_10K.sav', 'rb'))
		self.wordvec = pickle.load(open("word_vec", 'rb')) #Word2Vec(brown.sents())


	def compute_cluster(self, word):
		if word in self.wordvec.wv.vocab:
			return str(self.model.predict(self.wordvec[word].reshape(1,-1))[0] + 1)
		else:
			return 0

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    import os
    data_dir = 'data/lexicon/'
    feature_map = dict()
    fea_list = ['people.person.lastnames','firstname.5k','lastname.5000','people.person','people.family_name','tv.tv_program','tv.tv_network',
            'english.stop',
            'location.country','location','cvg.computer_videogame',
            'internet.website','business.consumer_product','product',
            'sports.sports_league','sports.sports_team'
                ]
    for fname in os.listdir(data_dir):
        fname_ = data_dir + '/' + fname
        if fname in fea_list:
            feature_map[fname] = set(line.strip().lower() for line in open(fname_))
    feature_dict = dict()
    for sent in train_sents:
        for word in sent:
            #if stop words
            word = word.lower()
            if word in feature_map['english.stop']:
                if word not in feature_dict:
                  feature_dict[word] = set()
                feature_dict[word].add("STOP_WORD")

            #NAME
            if word in feature_map['people.person.lastnames'] or word in feature_map['firstname.5k'] \
               or word in feature_map['lastname.5000'] or word in feature_map['people.person'] or word in feature_map['people.family_name']:
                if word not in feature_dict:
                  feature_dict[word] = set()
                feature_dict[word].add("NAME")

            #LOC
            if word in feature_map['location.country'] or word in feature_map['location']:
                if word not in feature_dict:
                  feature_dict[word] = set()
                feature_dict[word].add("LOC")

            #PROD
            if word in feature_map['business.consumer_product'] or word in feature_map['product']:
                if word not in feature_dict:
                  feature_dict[word] = set()
                feature_dict[word].add("PROD")

            # SPORTS
            if word in feature_map['sports.sports_league'] or word in feature_map['sports.sports_team']:
                if word not in feature_dict:
                    feature_dict[word] = set()
                feature_dict[word].add("SPORT")

            #WEB
            if word in feature_map['internet.website']:
                if word not in feature_dict:
                  feature_dict[word] = set()
                feature_dict[word].add("WEB")

            #VIDEO_GAME
            if word in feature_map['cvg.computer_videogame']:
                if word not in feature_dict:
                  feature_dict[word] = set()
                feature_dict[word].add("VID_GAME")

            #TV PGM
            if word in feature_map['tv.tv_program'] or word in feature_map['tv.tv_network']:
                if word not in feature_dict:
                  feature_dict[word] = set()
                feature_dict[word].add("TV")
    return feature_dict


def token2features(sent, i, cluster,feature_dict, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    if word.istitle():
      ftrs.append("IS_TITLE")

    flag = False
    if word.lower() in feature_dict:
        flag  = True
        for f in feature_dict[word.lower()]:
            ftrs.append(f)


    # adding prefix
    if len(word) > 3:
        ftrs.append("PRE_" + word[:3])
        # adding suffix
        ftrs.append("SUF_" + word[-3:])
    elif len(word) == 3:
        ftrs.append("PRE_" + word[:2])
        # adding suffix
        ftrs.append("SUF_" + word[-2])

    if word[0] == '#':
        flag = True
        ftrs.append("HASHTAG")
    if word[0] == '@':
        flag = True
        ftrs.append("MENTION")
    #
    # if flag == False:
    #     ftrs.append("OTHER")
    # getting from cluster
    ftrs.append('CLUSTER_' + str(int(cluster.compute_cluster(word))))

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, cluster,feature_dict, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, cluster,feature_dict, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    # sents = [
    # [ "Greg", "love", "Bronx", "India","Barada" ]
    # ]
    sents = [["@@twitterid" ,"john","is","halfway","#cleaneating","challenge","in","california" ]]
    feature_dict = preprocess_corpus(sents)
    cluster = Cluster()
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i,cluster, feature_dict)