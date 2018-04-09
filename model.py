import tensorflow as tf
import numpy as np
from keras.layers import GRU, Dense, Input, Bidirectional, concatenate, Layer, InputLayer
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K

class CustomEmbeddingLayer(InputLayer):

    def __init__(self, output_dim, embedding, **kwargs):
        self.output_dim = output_dim
        self.embedding = embedding
        self.embedding_len = embedding.shape[0]
        assert output_dim == embedding.shape[1]
        super(CustomEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomInputLayer, self).build(input_shape) 
        self.built = True

    def call(self, x):
        return K.dot(K.one_hot(K.argmax(x,axis=-1),self.embedding_len),self.embedding)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)


def simple_seq_to_seq_model(output_unit=None, hidden_unit=None, input_shape=None, output_seq_len=None,\
 decoder_input_unit=None,is_bidirect=True,batch_size=1,embedding=None):
    assert output_unit is not None
    assert output_seq_len is not None
    assert embedding is not None
    embedding_tensor = tf.convert_to_tensor(embedding,dtype=tf.float32)
    
    if input_shape is None:
        input_shape=(None,output_unit)
    if hidden_unit is None:
        hidden_unit= output_unit
    if decoder_input_unit is None:
        decoder_input_unit= embedding.shape[1]
    
    # input
    input_to_model=Input(shape=input_shape,batch_shape=(batch_size,)+input_shape, name="input_to_model")
    
    # encoder
    encoder= GRU(hidden_unit, name="encoder_GRU")
    h_t=encoder(input_to_model)

    # decoder
    dec_output=[]
    input_to_decoder=Input(shape=(1,decoder_input_unit), batch_shape=(batch_size,1,decoder_input_unit))
    input_=input_to_decoder
    
    decoder= GRU(output_unit,return_sequences=True, return_state=True, name="decoder_GRU")
    output_layer= Dense(output_unit, activation='softmax')
    embedding_lookup= CustomEmbeddingLayer(decoder_input_unit,embedding_tensor,batch_input_shape=(batch_size,1,decoder_input_unit))

    # loop over decoder states
    for i in range(output_seq_len):
        o_t,o_= decoder(input_,initial_state=h_t)
        o_t= output_layer(o_t)
        input_ = embedding_lookup(o_t)
        dec_output.append(o_t)
    
    o = concatenate(dec_output, axis=1)
    model = Model(inputs=[input_to_model,input_to_decoder],outputs=o)
    return model

