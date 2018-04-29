import tensorflow as tf
import numpy as np
from keras.layers import GRU, Dense, Input, Bidirectional, concatenate, Layer, InputLayer
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.constraints import max_norm


class AttentionLayer(Layer):
    def __init__(self, output_dim,**kwargs):
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("[LOG] input shape ", input_shape)
        self.W_h = self.add_weight((self.output_dim, self.output_dim),\
                                    name='W_h',\
                                    initializer="random_uniform",\
                                    regularizer=regularizers.l2(0.01),\
                                    constraint=max_norm(2.))
        
        self.W_a = self.add_weight((self.output_dim, self.output_dim),\
                                    name='W_a',\
                                    initializer="random_uniform",\
                                    regularizer=regularizers.l2(0.01),\
                                    constraint=max_norm(2.))
        self.V_a = self.add_weight((1,input_shape[1][1]),\
                                    name='V_a',\
                                    initializer="random_uniform",\
                                    regularizer=regularizers.l2(0.01),\
                                    constraint=max_norm(2.))
        
        super(AttentionLayer, self).build(input_shape) 
        self.built = True

    def call(self, x):
#         print("[LOG] call: input shape ", x[0].get_shape().as_list(), x[1].get_shape().as_list())
        x_=K.repeat_elements(K.expand_dims(x[0],axis=1),self.V_a.get_shape().as_list()[1],axis=1)
#         print("[LOG] call: x_ shape ", x_.get_shape())
        s_1=K.dot(x_,self.W_a)
#         print("[LOG] call: s_1 shape ", s_1.get_shape())
        s_2=K.dot(x[1],self.W_h)
#         print("[LOG] call: s_2 shape ", s_2.get_shape())
        s= s_1+s_2
        s=K.tanh(s)
        o=K.dot(self.V_a,s)
        o=K.squeeze(o,axis=0)
        alpha = K.softmax(o)
#         print("[LOG] call: alpha shape ", alpha.get_shape())
        delta=K.expand_dims(alpha,axis=1)
#         print("[LOG] call: delta shape ", delta.get_shape())
        gama= K.repeat_elements(delta,self.output_dim,axis=1)
#         print("[LOG] call: gamma shape ", gama.get_shape())
        beta= K.batch_dot(x[1],gama)
#         print("[LOG] call: beta shape ", beta.get_shape())
        return K.sum(beta,axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape
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
    encoder= GRU(hidden_unit,return_sequences=True, return_state=True, name="encoder_GRU")
    h_t,f_t=encoder(input_to_model)

    # decoder
    dec_output=[]
    input_to_decoder=Input(shape=(1,decoder_input_unit), batch_shape=(batch_size,1,decoder_input_unit))
    input_=input_to_decoder
    
    decoder= GRU(output_unit,return_sequences=True, return_state=True, name="decoder_GRU")
    output_layer= Dense(output_unit, activation='softmax')
    attention_initial= Dense(output_unit, activation='tanh')
    attention = AttentionLayer(output_unit)
    embedding_lookup= CustomEmbeddingLayer(decoder_input_unit,embedding_tensor,batch_input_shape=(batch_size,1,output_unit))

    s_t=attention_initial(f_t)
    
    # loop over decoder states
    for i in range(output_seq_len):
        c_t=attention([s_t,h_t])
#         print("[LOG] context dimension ",c_t.get_shape())
        o_t,o_= decoder(input_,initial_state=c_t)
        s_t=o_
        o_t= output_layer(o_t)
        input_ = embedding_lookup(o_t)
        dec_output.append(o_t)
    
    o = concatenate(dec_output, axis=1)
    model = Model(inputs=[input_to_model,input_to_decoder],outputs=o)
    return model
