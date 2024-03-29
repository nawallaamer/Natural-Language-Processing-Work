
##
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
import math
import numpy as np
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            print(word)
            print(freqDict[word])
            if freqDict[word] < 2:

                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement four kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using linear interpolation smoothing (SmoothedBigramModelInt)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,15):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.unigram_dist = UnigramDist(corpus)

    def generateSentence(self):
        sentence = [self.unigram_dist.draw() for _ in range(random.randint(10, 20))]
        return sentence

    def getSentenceProbability(self, sen):
        probability = 1.0
        for word in sen:
            probability *= self.unigram_dist.prob(word)
        return probability

    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        tot_words = 0
        for sent in corpus:
            tot_words += len(sent) - 2  # Exclude start and end tokens
            for word in sent:
                word_prob = self.unigram_dist.prob(word)
                if word_prob > 0:
                    log_prob += -1 * math.log(word_prob)
                else:
                    pass

        perplexity = math.exp(log_prob / tot_words)
        return perplexity
    #endddef
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.vocab = len(set(word for sen in corpus for word in sen))
        self.unigram_dist = UnigramDist(corpus)

    def generateSentence(self):
        sentence = [self.unigram_dist.draw() for _ in range(random.randint(10, 20))]
        return sentence

    def getSentenceProbability(self, sent):
        probability = 1.0
        for word in sent:
            probability *= (self.unigram_dist.counts[word] + 1) / (self.unigram_dist.total + self.vocab)
        return probability

    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        tot_words = 0
        for sent in corpus:
            tot_words += len(sent) - 2  # Exclude start and end tokens
            for word in sent:
                log_prob += -1 * math.log((self.unigram_dist.counts[word] + 1) / (self.unigram_dist.total + self.vocab))
        perplexity = math.exp(log_prob / tot_words)
        return perplexity
    #endddef
#endclass

# Unsmoothed bigram language model
# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.bigram_dist = BigramDist(corpus)

    def generateSentence(self):
        sentence = [start]
        current_word = start
        while current_word != end:
            next_word = self.bigram_dist.draw(current_word)
            sentence.append(next_word)
            current_word = next_word
        return sentence

    def getSentenceProbability(self, sent):
        prob = 1.0
        for i in range(1, len(sent)):
            prob = prob * math.exp(self.bigram_dist.prob(sent[i], sent[i-1]))
        return prob

    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        tot_words = 0
        for sent in corpus:
            tot_words += len(sent) - 2  # Exclude start and end tokens
            for i in range(1, len(sent)):
                log_prob += self.bigram_dist.prob(sent[i], sent[i-1])
        perplexity = math.exp(-log_prob / tot_words)
        return perplexity
    # enddef
# endclass

# Smoothed bigram language model (use linear interpolation for smoothing, set lambda1 = lambda2 = 0.5)

class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.unigram_dist = UnigramDist(corpus)
        self.bigram_dist = BigramDist(corpus)
        

    def generateSentence(self):
        sentence = [start]
        current_word = start
        while current_word != end:
            next_word = self.bigram_dist.draw(current_word)
            sentence.append(next_word)
            current_word = next_word
        return sentence

    def getSentenceProbability(self, sent):
        prob = 1.0
        for i in range(1, len(sent)):
            word = sent[i]
            previous_word = sent[i-1]
            unigram_prob = self.unigram_dist.prob(word)
            bigram_prob = self.bigram_dist.prob(word, previous_word)
            interpolated_prob = 0.5 * unigram_prob + 0.5 * (max(bigram_prob, 0.0) + 0.1) 
            prob = prob * interpolated_prob

        return prob

    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        tot_words = 0
        for sent in corpus:
            tot_words += len(sent) - 2  # Exclude start and end tokens
            for i in range(1, len(sent)):
                word = sent[i]
                previous_word = sent[i-1]
                unigram_prob = self.unigram_dist.prob(word)
                bigram_prob = self.bigram_dist.prob(word, previous_word)

                # Linear interpolation with add-k smoothing
                interpolated_prob = 0.5 * unigram_prob + 0.5 * (max(bigram_prob, 0.0) + 0.1)  # Add-k smoothing with k=0.1

                log_prob += math.log(interpolated_prob) if interpolated_prob > 0 else 0.0

        perplexity = math.exp(-log_prob / tot_words)
        return perplexity
    # enddef
# endclass



# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus=None):
        self.counts = defaultdict(float)
        self.total = 0.0
        if corpus:
            self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
        return UNK
	    #endif
	#endfor
    #enddef
#endclass

class BigramDist:
    def __init__(self, corpus):
        self.bicounts = defaultdict(float)
        self.unicounts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    def train(self, corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                if j!=0:
                    self.bicounts[(corpus[i][j],corpus[i][j-1])] += 1.0
                self.unicounts[corpus[i][j]] += 1.0
                self.total += 1.0

    def prob(self, word, givenword):
        if (word, givenword) in self.bicounts:
            return math.log(self.bicounts[word, givenword] / self.unicounts[givenword])
        else:
            return 0.0

    def draw(self,givenword):
        rand = self.unicounts[givenword]*random.random()
        for word1,word2 in self.bicounts:
            if word2 == givenword:
                rand -= self.bicounts[word1,word2]
                if rand <= 0.0:
                    return word1
        #
                    


#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    vocab = set()
    for sentence in trainCorpus:
        for word in sentence:
            if word in vocab:
                continue
            else:
                vocab.add(word)


    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    #Run sample unigram dist code
    unigramDist = UnigramDist(trainCorpus)
    print("Sample UnigramDist output:")
    print("Probability of \"picture\": ", unigramDist.prob("picture"))
    print("\"Random\" draw: ", unigramDist.draw())


    unigram_model = UnigramModel(trainCorpus)
    smoothed_unigram_model = SmoothedUnigramModel(trainCorpus)
    bigram_model = BigramModel(trainCorpus)
    smoothed_bigram_model = SmoothedBigramModelKN(trainCorpus)

    # Call methods for each model
    print("\nUnigram Model:")
    print("Generated Sentence:", unigram_model.generateSentence())
    print("Sentence Probability:", unigram_model.getSentenceProbability(["the", "horribly", "gruesome" ,"crimes", "that" ,"even", "the", "police", "surgeon", "can't", "stomach", "."]))
    print("Corpus Perplexity:", unigram_model.getCorpusPerplexity(trainCorpus))
    print("Positive Corpus Perplexity:", unigram_model.getCorpusPerplexity(posTestCorpus))
    print("Negative Corpus Perplexity:", unigram_model.getCorpusPerplexity(negTestCorpus))

    print("\nSmoothed Unigram Model:")
    print("Generated Sentence:", smoothed_unigram_model.generateSentence())
    print("Sentence Probability:", smoothed_unigram_model.getSentenceProbability(["the", "horribly", "gruesome" ,"crimes", "that" ,"even", "the", "police", "surgeon", "can't", "stomach", "."]))
    print("Corpus Perplexity:", smoothed_unigram_model.getCorpusPerplexity(trainCorpus))
    print("Positive Corpus Perplexity:", smoothed_unigram_model.getCorpusPerplexity(posTestCorpus))
    print("Negative Corpus Perplexity:", smoothed_unigram_model.getCorpusPerplexity(negTestCorpus))

    print("\nBigram Model:")
    print("Generated Sentence:", bigram_model.generateSentence())
    print("Sentence Probability:", bigram_model.getSentenceProbability(["the", "horribly", "gruesome" ,"crimes", "that" ,"even", "the", "police", "surgeon", "can't", "stomach", "."]))
    print("Corpus Perplexity:", bigram_model.getCorpusPerplexity(trainCorpus))
    print("PositiveCorpus Perplexity:", bigram_model.getCorpusPerplexity(posTestCorpus))
    print("Negative Corpus Perplexity:", bigram_model.getCorpusPerplexity(negTestCorpus))

    print("\nSmoothed Bigram Model:")
    print("Generated Sentence:", smoothed_bigram_model.generateSentence())
    print("Sentence Probability:", smoothed_bigram_model.getSentenceProbability(["the", "horribly", "gruesome" ,"crimes", "that" ,"even", "the", "police", "surgeon", "can't", "stomach", "."]))
    print("Corpus Perplexity:", smoothed_bigram_model.getCorpusPerplexity(trainCorpus))
    print("Positive Corpus Perplexity:", smoothed_bigram_model.getCorpusPerplexity(posTestCorpus))
    print("Negative Corpus Perplexity:", smoothed_bigram_model.getCorpusPerplexity(negTestCorpus))

    #Question Answers
    #Q1) Sentence Length is determined by rand loop as in this case it is set at 10 to 20 randomly where as in bigram sentence length depends when end token is predicted
    #Q2) Yes the probability differs as context gets involved in bigram and smoothing improves overall probabilty of both models
    #Q3) Bigram and smoothed bigram are producing more realistic sentences
    #Q4) Unigram Smoothed Unigram and bigram give training corpus higher perplexity where as smoothed bigram gives negative corpus higher probablity