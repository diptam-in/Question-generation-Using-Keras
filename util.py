# coding: utf-8
import logging as log
import tensorflow as tf
import os
import json as j
import re
import nltk
import operator
import math
import keras.backend as K
import numpy as np
from gensim.models import KeyedVectors
import sys  

log.basicConfig(level=log.INFO)

# for this, we are using squad data, it is given in json format
# this function loads the data and stores it in python dict
def load_data_into_dict(filename=None, fileloc="./"):
	if filename is None:
		return 
	file= open(fileloc+filename,"r",encoding='ascii')
	data= None
	try:
		data= j.load(file)
	except j.JSONDecodeError:
		log.error("Invalid Json Format")
	return data

# filter data accordingly
def filter_data(data=None, backup=False, n_samples=-1, backup_location="./backup/"):
	filtered_data_list=[]
	counter=0
	if n_samples < 0: 
		n_samples = len(data['data'])
	if data is None:
		return None
	if backup:
		if not os.path.exists(backup_location):
			os.makedirs(backup_location)
	for i in range(n_samples):
		for article in data['data'][i]['paragraphs']:
			qa_list=[]
			for qas in article['qas']:
				qa_list.append(tuple((qas['question'],qas['answers'][0]['text'],qas['answers'][0]['answer_start'])))
			filtered_data_list.append(tuple((article['context'],qa_list)))
			if backup:
				if not os.path.exists(backup_location+repr(counter)):
					os.makedirs(backup_location+repr(counter))
				f_write=open(backup_location+repr(counter)+"/context.txt","w")
				f_write.write(article['context'])
				f_write.close()
				f_write=open(backup_location+repr(counter)+"/qas.txt","w")
				f_write.write(repr(qa_list))
				f_write.close()
				counter+=1
	return filtered_data_list

# We need to split the text into smaller sentences!
# challenge is, proper splitting of sentences considering
# presence of abbreviation and symbols that may lead to wrong 
# splitting. o, instead of using nltk.tokenizer(), a separate
# approach is necessry. following code is adapted from 
# https://stackoverflow.com/questions/4576077/python-split-text-on-sentences

caps = "([A-Z])"
digits= "([0-9])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
removable = "(\"|,|\$|\(|\)|\?|\-|\--|\!|\")"

def replace_relevant_dots(text):
	text = " " + text + "  "
	text = text.replace("\n"," ")
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)
	text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	return text

def split_into_sentences(text):
	text= replace_relevant_dots(text)
	text = text.replace(".",". <stop>")
	text = text.replace("?","? <stop>")
	text = text.replace("!","! <stop>")
	text = text.replace("<prd>",".")
	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences

# utility function: for word tokenization
def custom_word_tokenizer(text):
	text= re.sub("(it's)|(It's)","it is",text)
	text= re.sub("I'm","I am",text)
	text= re.sub("I've","I have",text)
	text= re.sub("I'd","I would",text)
	text= re.sub("I'll","I will",text)
	text= re.sub("isn’t","is not",text)
	text= re.sub("hasn’t","has not",text)
	text= re.sub("(don’t)|(Don't)","do not",text)
	text= re.sub("(doesn’t)|(Doesn't)","does not",text)
	text= re.sub("(can’t)|(Can't)","can not",text)
	text= re.sub("couldn’t","could not",text)
	text= re.sub("aren’t","are not",text)
	text= re.sub("haven’t","have not",text)
	text= re.sub("(won’t)|(Won't)","will not",text)
	text= re.sub("wasn’t","was not",text)
	text= re.sub("(hadn’t)|(Hadn't)","had not",text)
	text= re.sub("(didn’t)|(Didn't)","did not",text)
	text= re.sub("(shouldn’t)|(Shouldn't)","should not",text)
	text= re.sub("(He’s)|(he's)","he is",text)
	text= re.sub("(Let’s)|(let's)","let us",text)
	text= re.sub("(She’d)|(she'd)","She would",text)
	text= re.sub("(what’s)|(What's)","what is",text)
	text= re.sub("(where’s)|(Where's)","Where is",text)
	text= re.sub("(You’d)|(you'd)","You would",text)
	text= re.sub("(He’d)|(he'd)","He would",text)
#     text= re.sub("\"","",text)
	text= re.sub("\-"," to ",text)
	text= re.sub(removable,"",text)
	text= replace_relevant_dots(text)
	splitted_list=(text).strip().split(" ")
	word_list=[]
	for i in range(len(splitted_list)):
#         word_list[i]= re.sub(removable,"",word_list[i])
		if splitted_list[i] is not None:
			temp=splitted_list[i].replace(".","").replace("<prd>",".").strip().lower() #NEED TO THINK
			if temp is not '':
				word_list.append(temp) 
	return word_list

# utility function: prepare pos-ne tags
def get_pos_tag(word_list):
	return nltk.pos_tag(word_list)
	
def get_ne_tag(tagged_pos):
	tree=nltk.ne_chunk(tagged_pos, binary=False)
	ne_list=[]
	for node in tree:
		if isinstance(node,tuple):
			ne_list.append(tuple((node[0],"NANE")))
		else:
			for ne in node:
				ne_list.append(tuple((ne[0],node.label())))
	return ne_list


# utility function: prepares bio tags
def get_bio_tag(text,answer,start_pos):
	bio_tag_list=[]
	first_slice= text[:start_pos]
	second_slice= text[start_pos:]
	second_slice= re.sub(answer,"<B>"+answer+"<E>",second_slice,count=1)
	text = first_slice + second_slice
	sent_list= split_into_sentences(text)
	o_flag=True
	i_flag=False
	for sent in sent_list:
		w_list= custom_word_tokenizer(sent)
		for word in w_list:
			if word.startswith("<B>"): o_flag=False
			if o_flag: bio_tag_list.append(tuple((word,"O")))
			elif not o_flag and i_flag: bio_tag_list.append(tuple((word.replace("<E>",""),"I")))
			else: bio_tag_list.append(tuple((word.replace("<B>",""),"B"))); i_flag=True
			if word.endswith("<E>"): o_flag=True
	return bio_tag_list            


# LSIT OF ALL POS_TAGS,NAMED_ENTITY_TAGS,BIO_TAGS used
ne_tag_list= tuple(["FACILITY", "GPE", "GSP", "LOCATION", "ORGANIZATION", "PERSON","NANE"])
pos_tag_list= tuple(['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS',\
'PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP',\
'WP$','WRB'])
bio_tag_list= tuple(['B','I','O'])
ne_tag_depth= len(ne_tag_list)
pos_tag_depth= len(pos_tag_list)
bio_tag_depth= len(bio_tag_list)


# utility function: assembles all tags in respective list
def get_tags(text,answer,start_pos):
	bio_tag=get_bio_tag(text,answer,start_pos)
	sentence_list= split_into_sentences(text)
	all_word=[];all_ne=[];all_pos=[];all_bio=[]
	counter=0
	for sentence in sentence_list:
		w_list= custom_word_tokenizer(sentence)
		t_pos= get_pos_tag(w_list)
		t_ne= get_ne_tag(t_pos)
		for pos,ne in zip(t_pos,t_ne):
			all_word.append(bio_tag[counter][0])
			all_bio.append(bio_tag_list.index(bio_tag[counter][1]))
			if pos[1] not in pos_tag_list: all_pos.append(18)
			else: all_pos.append(pos_tag_list.index(pos[1]))
			if ne[1] not in ne_tag_list: all_ne.append(6)
			else: all_ne.append(ne_tag_list.index(ne[1])) 
			counter+=1
	return all_word,all_bio,all_pos,all_ne

# this function encodes all lexical - positional features of a given text
# parms:
# text: text to encoded
# answer: answer string of the query to encode positional deature
# start_pos: start index of the answer
# return_word_list: If true, returns tokenized list of word; default False
# Return value: a tensor of shape (number_of_token,45)

def get_pos_lex_feat_embedding(text,answer,start_pos,padding,max_len,return_word_list=False):
	word,bio,pos,ne=get_tags(text,answer,start_pos)
	bio_one_hot= tf.one_hot(bio,bio_tag_depth,dtype=tf.float32)
	pos_one_hot= tf.one_hot(pos,pos_tag_depth,dtype=tf.float32)
	ne_one_hot= tf.one_hot(ne,ne_tag_depth,dtype=tf.float32)
	encoded_tag= tf.concat([bio_one_hot,pos_one_hot,ne_one_hot],axis=1)
	length=max_len-encoded_tag.get_shape().as_list()[0]
	pad=tf.zeros([length,encoded_tag.get_shape().as_list()[1]])
	if padding == "context": encoded = tf.concat([pad,encoded_tag],axis=0)
	else: encoded = tf.concat([encoded_tag,pad],axis=0)
	if return_word_list: return word,K.eval(encoded)
	else: return K.eval(encoded)

# utility function to prepare data dictionary
# param:
# data : a list of (context, qa_list)
#       context: text data i.e. article
#       qa_list: a list of (question, answer, start postion)
# max_vocab_length: number of words to retain sorted according to frquency
# retain_per: percentage of vocabulary to retain
# priroty is given to max_vocab_length, over retain_per
# return value: a tuple of words sorted according to frequency,
#               maximum sequence length for context, maximum sequence length for question  

def prepare_vocabulary(data, max_vocab_length=-1, retain_per=1.0, return_count=False):
	max_seq_length_context=0
	max_seq_length_question=0
	data_dict={}
	
	for context_qa_tup in data:
		context=custom_word_tokenizer(context_qa_tup[0])
		if max_seq_length_context<len(context): max_seq_length_context = len(context)
		for w in context:
			word = w.lower() # NEED TO THINK
			if data_dict.get(word) is None: data_dict[word]=0
			data_dict[word]+=1
			
		for qa in context_qa_tup[1]:
			question=custom_word_tokenizer(qa[0])
			if max_seq_length_question<len(question): max_seq_length_question = len(question)
			for w in question:
				word = w.lower() # NEED TO THINK
				if data_dict.get(word) is None: data_dict[word]=0
				data_dict[word]+=1
				
	# sort dictionary according to dictionary
	sorted_vocab=sorted(data_dict.items(), key=operator.itemgetter(1), reverse=True)
	if return_count: sorted_vocab_list=[x for x in sorted_vocab]
	else: sorted_vocab_list=[x[0] for x in sorted_vocab]
	if max_vocab_length<0:
			pos=math.ceil(len(sorted_vocab_list)*retain_per)
			return sorted_vocab_list[0:int(pos)],max_seq_length_context,max_seq_length_question
	else:
		if max_vocab_length < len(sorted_vocab_list):
			return sorted_vocab_list[0:max_vocab_length],max_seq_length_context,max_seq_length_question
		else: return sorted_vocab_list,max_seq_length_context,max_seq_length_question

# Prepares dictionary for word embedding
def get_index_word(sorted_vocab):
	vocabulary = dict()
	index=1
	for word in sorted_vocab:
		vocabulary[word] = index
		index=index+1
	return vocabulary

def get_one_hot_embedding_ques(question,vocabulary,max_len):
	size_of_vocab=len(vocabulary)+1
	word_list = custom_word_tokenizer(question)
	if len(word_list) > max_len: return None
	y_t=[]
	for word in word_list:
		if vocabulary.get(word) is None: y_t.append(0)
		else: y_t.append(vocabulary[word])
	y_ = tf.one_hot(y_t,size_of_vocab,dtype=tf.float32)
	length=max_len-y_.get_shape().as_list()[0]
	pad=tf.zeros([length,y_.get_shape().as_list()[1]])
	y=K.eval(tf.concat([y_,pad],axis=0))
	return y


# util function: input: emebdding file name,
#			vocabulary: a dictionary containing word-to-index as key-value pair
#		 output: loads embedding matrix for the required words, removes rest 
def get_embedding_matrix(embedding_file, vocabulary):
	EMBEDDING_FILE = embedding_file
	word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
	embedding_dim = 300
	embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
	embeddings[0] = 0  # So that the padding will be ignored
	# Build the embedding matrix
	for word, index in vocabulary.items():
		if word in word2vec.vocab:
			embeddings[index] = word2vec.word_vec(word)
	del word2vec
	return embeddings


# util function: input: list of word (essentially a sentence/article broken into words)
#                       padding: whether to pad for a sentence or for an article
#                output: a 2d numpy array consisting word-embedding of the given input

def get_word_embedding(text,padding,max_len_context,min_len_context,embeddings,vocabulary,max_len_question=100):  
	list_of_words = custom_word_tokenizer(text)
	print(len(list_of_words))
	if len(list_of_words) > max_len_context or len(list_of_words) < min_len_context:
		 return None
	word_vector=[]
	if padding=="context":
		for i in range(0,max_len_context-len(list_of_words)):
			word_vector.append(np.zeros(shape=embeddings[1].shape))
	for word in list_of_words:
		if vocabulary.get(word) is not None:
			word_vector.append(np.asarray(embeddings[vocabulary[word]]))
		else:
			word_vector.append(np.zeros(shape=embeddings[1].shape))
	if padding=="question":
		for i in range(0,max_len_question-len(list_of_words)):
			word_vector.append(np.zeros(shape=embeddings[1].shape))
	return np.asarray(word_vector)

