import tensorflow as tf
import numpy as np
from tqdm import tqdm

class BILSTM_CRF(object):
    def __init__(self,params=None,embedding_matrix=None):
        self.params=params
        self.embedding_matrix=embedding_matrix
        with tf.variable_scope("BILSTM_CRF"):
            self.build_input()
            self.add_embedding()
            self.add_forward()
            self.add_backward()
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_input(self):
        self.word_input=tf.placeholder(tf.int32,[None,self.params["max_sent_len"]])
        self.char_input=tf.placeholder(tf.int32,[None,self.params["max_sent_len"],self.params["max_word_len"]])
        self.tags=tf.placeholder(tf.int32,[None,self.params["max_sent_len"]])
        self.word_seq_len=tf.count_nonzero(self.word_input,axis=1,dtype=tf.int32)
        self.char_seq_len=tf.count_nonzero(self.char_input,axis=2,dtype=tf.int32)
        self.is_training=tf.placeholder(tf.bool)
        self.batch_size=tf.shape(self.word_input)[0]
        self.num_classes=len(self.params["tag2idx"])
        self.global_step = tf.Variable(0, trainable=False)

    def lstm_cell(self,rnn_size):
        return tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.orthogonal_initializer())

    def add_embedding(self):
        #if self.embedding_matrix:
        word_embedding=tf.Variable(initial_value=self.embedding_matrix,
                                name="embedding_matrix_w",
                                dtype="float32")
        # else:
        #     word_embedding = tf.get_variable("word_embedding",[len(self.params["word2idx"]), self.params["word_embed"]], tf.float32)

        char_embedding=tf.get_variable("char_embedding",[len(self.params["char2idx"]),self.params["char_embed"]],tf.float32)

        word_input=tf.nn.embedding_lookup(word_embedding,self.word_input)#(B,T,D_WORD)
        char_input=tf.nn.embedding_lookup(char_embedding,self.char_input)
        char_input=tf.reshape(char_input,[-1,self.params["max_word_len"],self.params["char_embed"]])#(B*T,MAX_CHAR,D_CHAR)

        _,(state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(self.params["char_hidden_size"]) for _ in range(self.params["n_layers"])]),
            cell_bw=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(self.params["char_hidden_size"]) for _ in range(self.params["n_layers"])]),
            inputs=char_input,
            sequence_length=tf.reshape(self.char_seq_len,[-1]),
            dtype=tf.float32,
            scope='bidirectional_rnn_char')

        final_state=tf.concat((state_fw[-1].h,state_bw[-1].h),-1)#(B*T,2*D_CHAR)
        final_char_state=tf.reshape(final_state,(-1,self.params["max_sent_len"],2*self.params["char_hidden_size"]))#(B,T,2*D_CHAR)
        self.final_embedding=tf.concat((word_input,final_char_state),-1)

    def add_forward(self):

        drop_embedding=tf.layers.dropout(self.final_embedding, self.params["dropout_rate"], training=self.is_training)

        (output_fw,output_bw), _= tf.nn.bidirectional_dynamic_rnn(
            cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(self.params["word_hidden_size"]) for _ in range(self.params["n_layers"])]),
            cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(self.params["word_hidden_size"]) for _ in range(self.params["n_layers"])]),
            inputs=drop_embedding,
            sequence_length=self.word_seq_len,
            dtype=tf.float32,
            scope='bidirectional_rnn_word')

        final_out=tf.concat((output_fw,output_bw),-1)#(B,T,2*D_WORD)
        tanh_layer=tf.layers.dense(final_out,self.params["word_hidden_size"],activation=tf.nn.tanh)
        self.training_logits=tf.layers.dense(tf.reshape(tanh_layer,[-1,self.params["word_hidden_size"]]),self.num_classes)#(B*T,CLASS)



    def add_backward(self):
        if not self.params["crf"]:
            # val_acc: 0.948 after 10 epoch
            masks=tf.sequence_mask(self.word_seq_len,self.params["max_sent_len"],dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=tf.reshape(self.training_logits,[self.batch_size,-1,self.num_classes]),
                                                         targets=self.tags,
                                                         weights=masks)
        else:
            # val_acc: 0.952 after 10 epoch
            logits=tf.reshape(self.training_logits,[self.batch_size,-1,self.num_classes])
            with tf.variable_scope('crf_loss'):
                self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    inputs= logits,
                    tag_indices=self.tags,
                    sequence_lengths=self.word_seq_len)
            #print("log_:", self.log_likelihood)#(64,)
            with tf.variable_scope('crf_loss', reuse=True):
                transition_params = tf.get_variable('transitions', [self.num_classes, self.num_classes])
            self.viterbi_out, _ = tf.contrib.crf.crf_decode(  # return the index of the highest scoring sequence of tags
                logits, transition_params, self.word_seq_len)
            #print("viterbi:",self.viterbi_out)#(64, 113)
            self.loss = tf.reduce_mean(-self.log_likelihood)

        self.learning_rate_exp = tf.train.exponential_decay(self.params["lr"], self.global_step,
                                                            self.params["decay_size"],
                                                            self.params["decay_factor"])
        params = tf.trainable_variables()
        optmizer = tf.train.AdamOptimizer(self.learning_rate_exp)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.params["grad_clip"])
        gard_op = optmizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        var_avg = tf.train.ExponentialMovingAverage(self.params["moving_average_decay"], self.global_step)
        variables_averages_op = var_avg.apply(params)
        with tf.control_dependencies([gard_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')

    def train(self,X_train,y_train,X_val,y_val,batch_size):
        total=len(X_train[0]) // batch_size + 1
        for epoch in range(self.params["num_epochs"]):
            for step,(word_input,char_input,tag_input) in enumerate(self.gen_batch(X_train,y_train,batch_size)):
                feed_dict={self.word_input:word_input,self.char_input:char_input,
                           self.tags:tag_input,self.is_training:True}
                loss,_=self.sess.run([self.loss,self.train_op],feed_dict=feed_dict)

                print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f"
                      % (epoch+1, self.params["num_epochs"], step + 1,total , loss))

                if step%100==0 and step!=0:
                    total_loss=0
                    total_acc=0
                    total_valid_len=0
                    counter=1
                    for (word_input, char_input, tag_input) in  tqdm(
                            self.gen_batch(X_val, y_val, batch_size),total=len(X_val)//batch_size + 1,ncols=70):
                        feed_dict = {self.word_input: word_input, self.char_input: char_input,
                                     self.tags: tag_input, self.is_training: False}

                        if not self.params["crf"]:
                            pred=tf.argmax(tf.reshape(self.training_logits, [self.batch_size, -1, self.num_classes]), -1)
                        else:
                            pred=self.viterbi_out

                        valid_len=tf.reduce_sum(self.word_seq_len)
                        val_loss,length,predict,word_seq_len = self.sess.run([self.loss,valid_len,pred,self.word_seq_len], feed_dict=feed_dict)
                        _pred,_tag_input=[],[]
                        for i in range(len(predict)):
                            _pred.append(predict[i][0:word_seq_len[i]])
                            _tag_input.append(tag_input[i][0:word_seq_len[i]])

                        _pred=np.array(_pred)
                        _tag_input=np.array(_tag_input)

                        _pred=np.array([x for pred in _pred for x in pred]).astype(dtype=np.int32)
                        _tag_input=np.array([x for tag in _tag_input for x in tag]).astype(dtype=np.int32)

                        accuracy=tf.reduce_sum(tf.cast(tf.equal(_pred,_tag_input),tf.float32))
                        acc=self.sess.run(accuracy,feed_dict=feed_dict)

                        total_acc += acc
                        total_valid_len += length
                        total_loss+=val_loss
                        counter+=1

                    print(" val_loss: %.3f | val_acc: %.3f"% (float(total_loss/counter),float(total_acc/total_valid_len)))



    #def predict(self,X_test,y_test):
    #this is the same as validation process

    def gen_batch(self,x,y,batch_size):
        for i in range(len(x[0])//batch_size+1):
            yield x[0][i*batch_size:(i+1)*batch_size],x[1][i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size]