import tensorflow as tf
import numpy as np
from config import args
from base_model import baseModel

class DecomposableNLIModel(baseModel):
    def __init__(self, embedding, num_class, embedding_size, vocab_size,
                 use_intra_attention=False, distance_biases=10):
        self.use_intra_attention=use_intra_attention
        self.distance_biases=10
        super(DecomposableNLIModel, self).__init__(embedding,num_class,
                                                   embedding_size,vocab_size)

    def input_transform(self, sent, reuse):
        projected=super(DecomposableNLIModel, self).input_transform(sent, reuse)
        if self.use_intra_attention:
            projected=self.get_intra_atten(projected,reuse)
        return projected

    def compare_transform(self, sent, scope, num_units, reuse):
        return self.MLP(sent, scope, num_units, reuse)

    def attend_transform(self, sent, scope, num_units, reuse):
        return self.MLP(sent, scope, num_units, reuse)

    def get_intra_atten(self,sent,reuse):
        T=tf.shape(sent)[1]
        with tf.variable_scope("intra_atten") as self.intra_scope:
            intra=self.MLP(sent,self.intra_scope,args.num_units,reuse)
            intra_atten=tf.matmul(intra,intra,transpose_b=True)#(B,T,T)
            bias=self.get_dis_bias(T,reuse)
            intra_atten+=bias
            intra_atten=self.softmax_atten(intra_atten)#maybe add masking here?
            intra_atten=tf.matmul(intra_atten,sent)
            output=tf.concat([sent,intra_atten],axis=-1)
            return output

    def get_dis_bias(self,time_steps,reuse):
        with tf.variable_scope('dist_bias', reuse):
            #distance_bias will be updated through bp
            distance_bias = tf.get_variable('dist_bias', [self.distance_biases],
                                            initializer=tf.zeros_initializer())  # (distance_biases,)

            # messy tensor manipulation for indexing the biases
            r = tf.range(0, time_steps)  # (time_steps,)
            r_matrix = tf.tile(tf.reshape(r, [1, -1]),  # (1,time_steps)
                               tf.stack([time_steps, 1]))  # (time_steps,time_steps)
            raw_inds = r_matrix - tf.reshape(r, [-1, 1])
            clipped_inds = tf.clip_by_value(raw_inds,1-self.distance_biases,self.distance_biases-1)
            clipped_inds=tf.abs(clipped_inds)
            values = tf.nn.embedding_lookup(distance_bias, clipped_inds)
        return values
