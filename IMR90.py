
from tensorflow.keras.layers import * 
#from tensorflow.keras.models import *
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import initializers
import itertools
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

from tensorflow.keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split


MAX_LEN_en = 3000
MAX_LEN_pr = 2000
#NB_WORDS = 4097
#EMBEDDING_DIM = 100
#embedding_matrix = np.load('embedding_matrix.npy')


class TransformerBlock(keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.rate = rate

        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout1 = keras.layers.Dropout(rate)
        self.layernorm1 = LayerNormalization()

        self.dense1 = Dense(dff, activation='relu')
        self.dropout2 = keras.layers.Dropout(rate)
        self.dense2 = Dense(d_model)
        self.dropout3 = keras.layers.Dropout(rate)
        self.layernorm2 = LayerNormalization()

    def call(self, inputs, training=True):
        attn_output, attn_weights = self.multi_head_attention(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        dense_output = self.dense1(out1)
        dense_output = self.dropout2(dense_output, training=training)

        dense_output = self.dense2(dense_output)

        dense_output = self.dropout3(dense_output, training=training)
        out2 = self.layernorm2(out1 + dense_output)

        return out2, attn_weights

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dff': self.dff,
            'rate': self.rate
        })
        return config



class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights2 = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(AttLayer, self).get_config()
        config['attention_dim'] = self.attention_dim
        return config






def get_model():
    enhancers=keras.layers.Input(shape=(MAX_LEN_en,4))
    promoters=keras.layers.Input(shape=(MAX_LEN_pr,4))

    
    enhancer_conv_layer = keras.layers.Conv1D(filters = 96,kernel_size = 40,padding = "valid",activation='relu')(enhancers)
    enhancer_max_pool_layer = keras.layers.MaxPooling1D(pool_size = 20, strides = 20)(enhancer_conv_layer)
    # enhancer_drop_layer=keras.layers.Dropout(0.5)(enhancer_max_pool_layer)
    
    promoter_conv_layer = keras.layers.Conv1D(filters = 96,kernel_size = 40,padding = "valid",activation='relu')(promoters)
    promoter_max_pool_layer = keras.layers.MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)
    # promoter_drop_layer=keras.layers.Dropout(0.5)(promoter_max_pool_layer)
   
    # merge_layer=keras.layers.Concatenate(axis=1)([promoter_drop_layer,enhancer_drop_layer])
    merge_layer=keras.layers.Concatenate(axis=1)([promoter_max_pool_layer,enhancer_max_pool_layer])

    output=keras.layers.BatchNormalization()(merge_layer)
    output=keras.layers.Dropout(0.5)( output)
    output, attn_weights=TransformerBlock(num_heads=16, d_model=96, dff=255, rate=0.5)(output)
    

    
    output = AttLayer(50)(output)
    #output=Flatten()(output)
    #output=Dense(925,activation='relu')(output)


    preds = keras.layers.Dense(1, activation='sigmoid')(output)   
    model=keras.models.Model([enhancers,promoters],preds)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
   


# In[ ]:


class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.en = val_data[0]
        self.pr = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.en,self.pr])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)

        self.model.save("./Model-IMR90/%sModel%d.tf" % (self.name,epoch))
     


       

        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return





t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all','all-NHEK']
name=names[0]
#The data used here is the sequence processed by data_processing.py.
Data_dir='./data/%s/'%name
train=np.load(Data_dir+'%s_train.npz'%name)
#test=np.load(Data_dir+'%s_test.npz'%name)
X_en_tra,X_pr_tra,y_tra=train['X_en_tra'],train['X_pr_tra'],train['y_tra']
#X_en_tes,X_pr_tes,y_tes=test['X_en_tes'],test['X_pr_tes'],test['y_tes']
print(X_en_tra.shape)
print(X_en_tra[-1])
print(X_pr_tra.shape)
print(y_tra.shape)
#y_tra=np.append(y_tra,0.0)
#print(y_tra)
X_en_tra, X_en_val,X_pr_tra,X_pr_val, y_tra, y_val=train_test_split(
    X_en_tra,X_pr_tra,y_tra,test_size=0.05,stratify=y_tra,random_state=250)


model=None
model=get_model()
model.summary()
print ('Traing %s cell line specific model ...'%name)


back = roc_callback(val_data=[X_en_val, X_pr_val, y_val], name=name)
history=model.fit([X_en_tra, X_pr_tra], y_tra, validation_data=([X_en_val, X_pr_val], y_val), epochs=300, batch_size=64,
                  callbacks=[back])

t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("????:"+t1+"????:"+t2)
















