# coding: utf-8
from util import *
from model import simple_seq_to_seq_model
import numpy as np
import tensorflow as tf


def prepare_train_data(data,embedding,vocabulary,max_context=1000,max_question=1000,min_context=0,min_question=0,number_of_sample=100):
	x_train=[]
	y_train=[]
	flag=False
	size_of_vocab=len(vocabulary)+1
	for context_qa_tup in data:
		context = context_qa_tup[0]
		question_list = context_qa_tup[1]
		context_word_embedding = get_word_embedding(context,"context",max_context,min_context,embedding,vocabulary)
		if context_word_embedding is not None:
			for question_info in question_list:
				question = question_info[0]
				answer = question_info[1]
				position = question_info[2]
				context_tag_embedding = get_pos_lex_feat_embedding(context,answer,position,"context,",max_context)
				context_embedding = np.concatenate([context_word_embedding,context_tag_embedding],axis=1)
				temp=get_one_hot_embedding_ques(question,vocabulary,max_question)
				if temp is not None:
					x_train.append(context_embedding)
					y_train.append(temp)
					number_of_sample-=1
					if number_of_sample == 0:
						flag=True
						break
			if flag: break

	print("[LOG] data preparation done")
	x_ = np.dstack(x_train)
	x = np.rollaxis(x_,-1)
	ytr = np.dstack(y_train)
	y= np.rollaxis(ytr,-1)
	return x,y


def main():
	squad_file_name="dev-v1.1.json"
	squad_file_location="./squad/"
	parsed_backup_location="./backup/"
	embedding_file="./GoogleNews-vectors-negative300.bin.gz"
	number_of_context=2
	batch_size=5
	epochs=2
	n_sample=10

	data = load_data_into_dict(filename=squad_file_name,fileloc=squad_file_location)
	data = filter_data(data,n_samples = number_of_context)
	list_of_word,max_context,max_question = prepare_vocabulary(data, retain_per=0.9)
	vocabulary = get_index_word(list_of_word)
	embedding = np.random.random((len(list_of_word)+1,300))

	x_train,y_train=prepare_train_data		(data,embedding,vocabulary,max_context=150,max_question=30,min_context=100,min_question=20,number_of_sample=n_sample)

	
	number_of_input=x_train.shape[0]-(x_train.shape[0]%5)
	token_length=x_train.shape[2]
	sequence_length=x_train.shape[1]
	output_seq_len=y_train.shape[1]
	hidden_unit=len(vocabulary)+1
	decoder_ip_unit=embedding.shape[1]
	output_dimesion=len(vocabulary)+1
	y_initial= np.ones([number_of_input,1,decoder_ip_unit])

	print("----------- CONFIGURATION --------------")
	print("batch:",batch_size)
	print("number of input:",number_of_input)
	print("word embedding length:",token_length)
	print("input sequnce length:",sequence_length)
	print("output sequnce length:",output_seq_len)
	print("hidden unit:",hidden_unit)
	print("output dimension:",output_dimesion)
	print("epoch:",epochs)

	model=simple_seq_to_seq_model(output_unit=output_dimesion,input_shape=(sequence_length,token_length),hidden_unit=hidden_unit,\
		decoder_input_unit=decoder_ip_unit, output_seq_len=output_seq_len,batch_size=batch_size, embedding=embedding)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	print("[LOG] Model compiled")
	history=model.fit([x_train,y_initial],y_train,epochs=epochs,batch_size=batch_size)


if __name__ == "__main__":
	main()

