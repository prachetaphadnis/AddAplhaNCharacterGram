from pprint import pprint as p
import numpy as np
import string
import itertools
from collections import OrderedDict
import re
import random
import argparse
				
V=30                                                                    #vocabulary size
vocab=list(string.ascii_lowercase) 										# vocabulary
vocab+=['0','.',' ','#']

possible_trigrams=[''.join(i) for i in list(itertools.product(vocab, repeat=3))]   #all seen plus unseen trigrams
				  
counts={}	#stores bigram and trigram counts
'''
processes a string according to language rules
inputs: a string of characters
output: processed string
'''
def  preprocess_line(string):
	string=re.sub(r"[^A-Za-z0-9\.\s\']","",string)		#removing any character that is not alphabet,digit,'.', apostrophe or space
	string=re.sub(r"\'\s+","",string)					#remove "' "
	string=re.sub(r"\'","",string)						#remove "'"
	string=re.sub(r"[0-9]","0",string)					#replacing all digits with 0
	string=re.sub(r"[\s][\s]+"," ",string)				#replacing multiple spaces with single space
	string=string.lower()								#lowercasing
	string="##"+string+"#"								# adding sentence delimiters									
	return string

'''
Reads a text file and splits by sentence into a corpus
inputs: file:file name
		corpus:empty corpus
output: filled corpus
'''
def read_file(file,corpus):
	file=open(file,"r")													#open the training file to read
	full_text="".join([line for line in file])							# joining all lines in the file to one string
	corpus=full_text.split("\n")[:-1]										# splitting the lines in the string on occurence of newline \n character and storing in a list
	return corpus
 	
'''
Preprocesses each line of corpus
input: corpus=filled corpus
output: preprocessed corpus
'''
def preprocess_corpus(corpus):
	for i in range(len(corpus)):
		corpus[i]=preprocess_line(corpus[i])										# replacing each sentence in corpus with preprocessed sentence
	return corpus	

# counts a substring in the corpus
# inputs: corpus and a string(bigram or trigram)
# output: count of that gram in the corpus
def get_count(corpus,gram):

	if gram not in counts:
		counts[gram]=(''.join(corpus)).count(gram)
	return counts[gram]

'''
calculates perplexity
inputs: model:Language model
		test_data: testing corpus
		alpha: alpha value for smoothing
output pp:perplexity of the testing corpus wrt model
'''
def get_pp(model,test_data,alpha):
	
	avg_log_prob_sequence=[]
	N=0
	for i in range(len(test_data)):
		line=test_data[i]
		prob=0
		N=0
		for j in range(2,len(line)):
			trigram=line[j-2]+line[j-1]+line[j]
			bigram=line[j-2]+line[j-1]
			if trigram in model:
				prob-=np.log2(model[trigram])
		
		N=len(line)                                             # average out over line first for trigrams
		avg_log_prob_sequence.append(prob/N)                    #average log probabilities for each sentence sequnce
   
	avg_log_prob_total=np.sum(avg_log_prob_sequence)/len(test_data)     #average log probabilities for the enitre document
	
	pp=np.power(2,avg_log_prob_total)
	return pp

'''
returns an add-alpha model for only seen trigrams in a given corpus
inputs:	corpus:filled corpus
		model:language model
		alpha: alpha value for smoothing
output: changes done in model itself
'''
def add_alpha_seen(corpus,model,alpha):
	
	
	for line in corpus:
		t_count=0
		b_count=0
		for i in range(2,len(line)):								#skipping the first two ##
			trigram=line[i-2]+line[i-1]+line[i]						
			if( trigram not in model):								#checking if the trigram is already been recorded before
				bigram=line[i-2]+line[i-1]
				t_count=get_count(corpus,trigram)
				b_count=get_count(corpus,bigram)
				model[trigram]=float(float(t_count+alpha)/float(b_count+alpha*V)) #add-alpha smoothing
	
'''
returns an add-alpha model for only all possible trigrams in a given corpus
inputs:	corpus:filled corpus
		model:language model
		alpha: alpha value for smoothing
		possible_trigrams: a list of all possible trigrams for our vocabulary
output: changes done in model itself
'''
def add_alpa_unseen_MLE(corpus,model,alpha,possible_trigrams):
	
	for trigram in possible_trigrams:    				#		
		
		if( trigram not in model):								#skipping already seen trigrams
			bigram=trigram[0:2]
			b_count=0
			t_count=0
			b_count=get_count(corpus,bigram)
			model[trigram]=float((t_count+alpha)/(b_count+alpha*V)) #add-alpha smoothing
	
'''
finds the optimum alpha for smoothing our model from a list of alphas
inputs: train_corpus: 80% of our training data
		dev_corpus: 20% of our training data
outputs: optimum alpha
'''
def best_alpha(train_corpus,dev_corpus):                          #optimizing alpha to be one that minimizes perplexity of the dev set
	model={}
	alpha_options=np.arange(0.1,1.0,0.1)
	alpha_perplexity={}
	for i in alpha_options:
		add_alpha_seen(train_corpus,model,i)                             #passing alpha to the model for train set
		alpha_perplexity[i]=get_pp(model,dev_corpus,i)                  #getting perplexity for that alpha for dev set
	return min(alpha_perplexity,key=alpha_perplexity.get)           #get argmin PP(alpha)
	
'''
trains language model
inputs: file: file name of training data
outputs: smooth_model: model trained using add-alpha smoothing
		 optimum alpha: alpha value used for smoothing
'''
def build(file):

	corpus=[]
	smooth_model=OrderedDict()
	corpus=read_file(file,corpus)
	corpus= preprocess_corpus(corpus)
	train_corpus=corpus[:len(corpus)*80//100]                               #dividing training set to 80percent training and 20percent development set in order to choose optimal alpha
	dev_corpus=corpus[(len(corpus)*80//100):]
	opt_alpha=best_alpha(train_corpus,dev_corpus)
	
	add_alpha_seen(corpus,smooth_model,opt_alpha)

	add_alpa_unseen_MLE(corpus,smooth_model,opt_alpha,possible_trigrams)
	
	file_name='smooth_model_'+file[-2:]
	print(str(file[-2:])," Model trained! Saving in file: "+str(file_name))
	f = open(file_name,'w')                   # writing model to file
	for k in sorted(smooth_model):
	   f.write(k+"\t"+str(smooth_model[k])+"\n")

	return smooth_model,opt_alpha

def get_unigram_probs(model,bigram):              
	uni_probs={}
	matches=[key for key in model.keys() if key.startswith(bigram)]     #Finds the all trigrams that start with the given bigram
	for match in matches:
		uni_probs[match[-1]]=model[match]   #Assigns the trigram probability as probability of the unigram given the bigram
	
	return list(uni_probs.keys()),list(uni_probs.values())

def gen_character(keys,values,length):          

	total=sum(values)                                                   
	values=[value/total for value in values]
	bins = np.cumsum(values)                    #binning, bin[i] gives the upper bound of the bi
	random_value=random.uniform(0,1)        # generate a random value from 0 to 1
	low=0
	high=len(keys)-1
	while (high >= low):
		mid = (low + high) // 2;            
		
		if ( bins[mid] < random_value):         # if number generated is greater than mid bin value, then search in right half
			
			low = mid + 1
		elif ( bins[mid] - values[mid] > random_value): #if number generated is lower than the entire middle bin, search in left half
			
			high = mid - 1
		else:                                           # number generated falls in bin[mid-1]-bin[mid]
			return keys[mid]

def load_saved_model(file_name):
	model={}
	file=open(file_name,"r")
	for line in file:
		model[line.split("\t")[0]]=float(line.split("\t")[1])	#reading each line and splitting trigrams and values as key,value for the model.

	return model
'''
generates a weighted random sequence for a language model
input: language model, sequence length
output: generated sequence
'''
def generate(model,length):
	sequence="##"                           #Assumption: Sentence begins with ## and ends with #
	l=2
	for i in range(2,length-1):             #Sequence begins from l=2 since ## is needed there
		keys,values=get_unigram_probs(model,sequence[-2]+sequence[-1])            #gets a unigram model given bigram
		c=gen_character(keys,values,1)
		sequence+=c
		if c=='#':
			sequence+='\n##'
	sequence+="#"
	print(sequence)

def test_perplexity(models,test):
    
    test_corpus=[]
    test_corpus=preprocess_corpus(read_file("test",test_corpus))        #reading in and splitting the test corpus by sentences.
    perplexity={}
    
    

    for language in models:
        print("\nTesting ",language," model on test file ")
        model=models[language]["model"]
        opt_alpha=models[language]["alpha"]
        smooth_model_perplexity=get_pp(model,test_corpus,opt_alpha)        #get perplexity ofthe model
        perplexity[language]=smooth_model_perplexity
    print("\n\nTEST RESULTS:")
    
    for language in perplexity:
        print("Perplexity for ",language,": ",perplexity[language])
    print("Predicted language for test file: ", min(perplexity, key=perplexity.get))
	
def main(files):
		
	#files=["training.en","training.de","training.es"]						#	training file path (relative in this case,can be absolute)
	models={}
	for file in files:
		print("\nTraining ", str(file[-2:]), "Model...")
		global counts
		counts={}
		models[file[-2:]]={}
		model,alpha=build(file)
		models[file[-2:]]["model"]=model
		models[file[-2:]]["alpha"]=alpha
			   
	return models   



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Add Alpha N-Character Gram')
	parser.add_argument('files', metavar='N', type=str, nargs='+',
                   help='training file or list of files. format = filename.<2 letter lang> example: training.en for english' )
	parser.add_argument('test', metavar='T', type=str, nargs=1,
                   help='test file')
	args = parser.parse_args()
	#print(args.files)
	models=main(args.files)
	test_perplexity(models,args.test)
	
