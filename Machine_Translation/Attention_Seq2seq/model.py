import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as core_layers
from prepare_data2 import preprocess
import re
import os
class NMT(object):
    def __init__(self,X_word2idx,X_idx2word,Y_word2idx,Y_idx2word,rnn_size=100,n_layers=2,grad_clip=5.0,
                embedding_size=350,beam_width=5,force_teaching_ratio=0.5,learning_rate=0.0001,decay_size=1000,
                 decay_factor=0.99,moving_average_decay=0.99,save_path="./checkpoint/"):
        self.X_word2idx=X_word2idx
        self.Y_word2idx=Y_word2idx
        self.X_idx2word=X_idx2word
        self.Y_idx2word=Y_idx2word
        self.rnn_size=rnn_size
        self.n_layers=n_layers
        self.force_teaching_ratio = force_teaching_ratio
        self.embedding_size=embedding_size
        self.beam_width=beam_width
        self.learning_rate=learning_rate
        self.decay_factor=decay_factor
        self.decay_size=decay_size
        self.grad_clip=grad_clip
        self.save_path=save_path
        self.moving_average_decay=moving_average_decay
        self.register_symbols()
        self.build_graph()


    def build_graph(self):
        self.setup_input()
        self.add_encoder_layer()
        with tf.variable_scope("decoder"):
            self.add_training_decoder()
        with tf.variable_scope("decoder",reuse=True):
            self.add_infer_decoder()
        self.loss()
        self.initialize()


    def setup_input(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.target=tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.shape(self.X)[0]


    def lstm_cell(self, rnn_size=None, reuse=False):
        rnn_size = self.rnn_size if rnn_size is None else rnn_size
        return tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)

    # end method


    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.X_word2idx), self.embedding_size],
                                            tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        self.encoder_out = tf.nn.embedding_lookup(encoder_embedding, self.X)
        self.encoder_state= tuple()
        for n in range(self.n_layers):
            #A tuple (output_state_fw, output_state_bw) containing the forward and the backward final states of bidirectional rnn.
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.lstm_cell(self.rnn_size//2 ),
                cell_bw=self.lstm_cell(self.rnn_size//2 ),
                inputs=self.encoder_out,
                sequence_length=self.X_seq_len,
                dtype=tf.float32,
                scope='bidirectional_rnn_' + str(n))
            self.encoder_out = tf.concat((out_fw, out_bw), 2)

        print("state_fw:",state_fw.c)#state_fw: Tensor("bidirectional_rnn_1/fw/fw/while/Exit_2:0", shape=(?, 50), dtype=float32
        bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
        print("bi_state_c:", bi_state_c)#bi_state_c: Tensor("concat_2:0", shape=(?, 100), dtype=float32)
        bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
        self.encoder_state = tuple([bi_lstm_state] * self.n_layers)


    def add_training_decoder(self):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.rnn_size,
            memory=self.encoder_out,
            memory_sequence_length=self.X_seq_len)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.rnn_size)
        decoder_embedding = tf.get_variable('decoder_embedding', [len(self.Y_word2idx), self.embedding_size],
                                            tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=tf.nn.embedding_lookup(decoder_embedding,self.Y),
            sequence_length=self.Y_seq_len,
            embedding=decoder_embedding,
            sampling_probability=1 - self.force_teaching_ratio,
            time_major=False
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            helper=training_helper,
            initial_state=self.decoder_cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=self.encoder_state),
            output_layer=core_layers.Dense(len(self.Y_word2idx))
        )
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.Y_seq_len))
        self.training_logits = training_decoder_output.rnn_output#(100, 29, 8722)

    def add_infer_decoder(self):
        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]

        self.encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.beam_width)
        self.encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.beam_width)
        self.X_seq_len_tiled = tf.contrib.seq2seq.tile_batch(self.X_seq_len, self.beam_width)

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.rnn_size,
            memory=self.encoder_out_tiled,
            memory_sequence_length=self.X_seq_len_tiled)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse=True) for _ in range(self.n_layers)]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.rnn_size)

        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=tf.get_variable('decoder_embedding'),
            start_tokens=tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
            end_token=self._y_eos,
            initial_state=self.decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(
                cell_state=self.encoder_state_tiled),
            beam_width=self.beam_width,
            output_layer=core_layers.Dense(len(self.Y_word2idx), _reuse=True),
            length_penalty_weight=0.0)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=False,
            maximum_iterations=2 * tf.reduce_max(self.X_seq_len))
        self.predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]#shape=(?, ?, 5)==>(batch,seq_len,beam_size),这里0代表每个time step中beam得分最高的id
        #predicting_decoder_output: FinalBeamSearchDecoderOutput(predicted_ids=<tf.Tensor 'decoder_1/decoder/transpose:0' shape=(?, ?, 5) dtype=int32>, beam_search_decoder_output=BeamSearchDecoderOutput(scores=<tf.Tensor 'decoder_1/decoder/transpose_1:0' shape=(?, ?, 5) dtype=float32>, predicted_ids=<tf.Tensor 'decoder_1/decoder/transpose_2:0' shape=(?, ?, 5) dtype=int32>, parent_ids=<tf.Tensor 'decoder_1/decoder/transpose_3:0' shape=(?, ?, 5) dtype=int32>))
        #print("predicting_decoder_output:",predicting_decoder_output)


    def loss(self):
        self.global_step=tf.Variable(0,trainable=False)
        self.learning_rate_exp = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_size,
                                                            self.decay_factor)

        #more easier way is to use tf.to_float(tf.sign(self.target)) as masks directly
        self.masks=tf.sequence_mask(self.Y_seq_len,tf.reduce_max(self.Y_seq_len),dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                     targets=self.target,
                                                     weights=self.masks)
        params = tf.trainable_variables()
        optmizer = tf.train.AdamOptimizer(self.learning_rate_exp)
        # grad_vars=optmizer.compute_gradients(self.loss,params)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_clip)
        gard_op = optmizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        var_avg = tf.train.ExponentialMovingAverage(self.moving_average_decay, self.global_step)
        variables_averages_op = var_avg.apply(params)
        self.saver = tf.train.Saver(tf.trainable_variables())
        with tf.control_dependencies([gard_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')


    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver(tf.trainable_variables())

    def save_model(self,path):
        self.saver.save(self.sess, save_path=path)

    def load_model(self,path):
        self.saver.restore(sess=self.sess, save_path=path)

    def train(self,x_train,y_train,x_val,y_val,epoch=None,batch_size=100,file_id=None,file_nums=None):
        #print(len(x_val),len(y_val))
        #print(x_val,y_val)
        if os.path.exists(self.save_path):
            print("restoring model....")
            self.load_model(path=self.save_path+"model.ckpt")

        x_val_batch,in_val_batch,target_val_batch,x_val_len,y_val_len=next(self.gen_batch(x_val,y_val,batch_size))
        for step,(x_batch,in_batch,target_batch,x_len,y_len) in enumerate(self.gen_batch(x_train,y_train,batch_size)):
            loss,_,lr=self.sess.run([self.loss,self.train_op,self.learning_rate_exp],feed_dict={self.X:x_batch,self.Y:in_batch,self.target:target_batch,
                                                                      self.X_seq_len:x_len,self.Y_seq_len:y_len})

            print("Epoch %d/%d | file_split %d/%d |Batch %d/%d | train_loss: %.3f| lr: %.8f"
                  % (epoch, 10, file_id,file_nums,step+1, len(x_train) // batch_size, loss,lr))

            if step%500==0:
                val_loss = self.sess.run(self.loss, feed_dict={self.X: x_val_batch, self.Y: in_val_batch,
                                                               self.target: target_val_batch,
                                                               self.X_seq_len: x_val_len, self.Y_seq_len: y_val_len})
                print("val_loss: %.3f" % (val_loss))
                #self.infer("Deep Learning is very useful for solving some problems. ")
                self.infer("I love you. ",load_model=False)

            if step%1000==0 and step !=0:
                print("saving model......")
                self.save_model(path=self.save_path+"model.ckpt")

    def infer(self,line,load_model=True):
        if load_model:
            self.load_model(path=self.save_path+"model.ckpt")
        self.Y_idx2word[-1]="-1"
        line=preprocess(line)
        line = re.sub(r"\s+", " ", line)  # Akara is    handsome --> Akara is handsome
        line = line.strip().lower()  # lowercase
        input_idx=[self.X_word2idx.get(char,self._x_unk) for char in line.split()]
        out_idx = self.sess.run(self.predicting_ids, {
            self.X: [input_idx], self.X_seq_len: [len(input_idx)]})[0]
        out_line="".join([self.Y_idx2word.get(x) for x in out_idx])
        print("Encoder:",line)
        print("Decoder:",out_line)
        return out_line


    def pad(self,batch,pad):
        decoder_batch=[]
        in_batch = []
        target_batch=[]
        seq_len=[]
        max_len=max([len(x) for x in batch])
        if pad=="Y":
            for x in batch:
                new_in=[self._y_go]+ x + [self._y_pad] * (max_len - len(x))
                new_out=x + [self._y_eos] + [self._y_pad] * (max_len - len(x))
                in_batch.append(new_in)
                target_batch.append(new_out)
                seq_len.append(len(x)+1)
        else:
            for x in batch:
                decoder_batch.append(x + [self._x_pad] * (max_len - len(x)))
                seq_len.append(len(x))

        return decoder_batch,in_batch,target_batch,seq_len

    def gen_batch(self,x,y,size):
        #print((int(len(x)/self.batch_size))+1)
        for i in range(0,(int(len(x)/size))):
            x_batch=x[i*size:(i+1)*size]
            y_batch=y[i*size:(i+1)*size]
            decoder_batch_padded,_,_,seq_len_x=self.pad(x_batch,pad="X")
            _,in_batch_padded,target_batch_padded,seq_len_y=self.pad(y_batch,pad="Y")
            yield  decoder_batch_padded,in_batch_padded,target_batch_padded,seq_len_x,seq_len_y


    def preprocess_decoder_input(self):
        decoder_input=tf.strided_slice(self.Y,[0,0],[self.batch_size,-1],[1,1])
        #decoder_input=tf.concat([tf.fill([self.batch_size,1],self._y_go),decoder_input],1)
        return  decoder_input

    def preprocess_decoder_output(self):
        decoder_output=tf.strided_slice(self.Y,[0,1],[self.batch_size,tf.shape(self.Y)[1]],[1,1])
        #decoder_output=tf.concat([tf.fill([self.batch_size,1],tf.shape(self.Y)[1]),decoder_output],1)
        return decoder_output


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = self.X_word2idx['<PAD>']
        self._x_unk = self.X_word2idx['<UNK>']

        self._y_go = self.Y_word2idx['<GO>']
        self._y_eos = self.Y_word2idx['<EOS>']
        self._y_pad = self.Y_word2idx['<PAD>']
        self._y_unk = self.Y_word2idx['<UNK>']