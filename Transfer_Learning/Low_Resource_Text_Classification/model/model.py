import tensorflow as tf
from config import args
from tools import eval_confusion_matrix

_no_value = object()

class Model(object):
    def __init__(self,logger,vocab_size,embedding=_no_value):
        self.logger=logger
        self.vocab_size=vocab_size
        self.embedding=embedding


    def get_embedding(self,input,reuse=False):
        with tf.variable_scope("embedding",reuse=reuse):
            if self.embedding is _no_value:
                embedding = tf.get_variable("embedding_matrix",[self.vocab_size, args.embedding_size], tf.float32)
            else:
                embedding = tf.Variable(initial_value=self.embedding,
                                             name="embedding_matrix",
                                             dtype="float32")
            embed = tf.nn.embedding_lookup(embedding, input)
            cnn_inputs = tf.expand_dims(embed, -1)#(B,W,H,1)
            # Input dropout
            cnn_inputs = tf.nn.dropout(cnn_inputs, keep_prob=args.dropout)
        return  cnn_inputs

    def lstm_cell(self,rnn_size):
        return tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.orthogonal_initializer())

    def cnn_module(self,input,reuse=False):
        with tf.variable_scope("cnn_module",reuse=reuse):
            cnn_input=self.get_embedding(input)
            filter_sizes=args.filter_sizes.split(",")
            filter_sizes=[int(x) for x in filter_sizes]
            max_feat_len=args.max_len-max(filter_sizes)+1
            #total_filters = args.num_filters * len(filter_sizes)
            output=[]
            for i, kernel_w in enumerate(filter_sizes):
                kernel = [kernel_w, args.embedding_size]
                # [B,seq-kw+1,1,num_filters]
                conv_out = tf.layers.conv2d(inputs=cnn_input, filters=args.num_filters,
                                            padding="valid", kernel_size=kernel, activation=tf.nn.relu)
                if args.clf=="cnn":
                    pool_out = tf.layers.max_pooling2d(inputs=conv_out, pool_size=[args.max_len - kernel_w + 1, 1],
                                                       strides=1)
                    pool_out=tf.squeeze(pool_out)
                    output.append(conv_out)
                elif args.clf=="clstm":
                    conv_out=tf.squeeze(conv_out,[2])
                    conv_out=conv_out[:,:max_feat_len,:]
                    output.append(conv_out)

            if len(filter_sizes)>1:
                cnn_out=tf.concat(output,-1)
            else:
                cnn_out=output
        return cnn_out

    def lstm_module(self,input,reuse=False):
        with tf.variable_scope("lstm_module", reuse=reuse):
            cells=[self.lstm_cell(args.hidden_size) for _ in range(args.nlayers)]
            cells=[tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=args.dropout) for cell in cells]
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.initial_state = cells.zero_state(args.batch_size, tf.float32)
            _, state = tf.nn.dynamic_rnn(cells, input, self.seq_len, self.initial_state)
        return state[-1].h

    def forward(self,features,mode):
        answer,question=features["answer"],features["question"]
        self.seq_len=self.get_seq_len(answer,question)
        if args.clf=="cnn":
            cnn_ans = self.cnn_module(answer)
            cnn_ques = self.cnn_module(question, reuse=True)
            cnn_output = tf.concat([cnn_ans, cnn_ques], -1)
            hidden=cnn_output
        elif args.clf=="lstm":
            ans_emb=self.get_embedding(answer)
            ques_emb=self.get_embedding(question,reuse=True)
            lstm_input=tf.concat([ans_emb,ques_emb],-1)
            hidden=self.lstm_module(lstm_input)
        elif args.clf=="clstm":
            if args.pretrain:
                cnn_output = self.cnn_module(answer)
            else:
                cnn_ans = self.cnn_module(answer)
                cnn_ques = self.cnn_module(question, reuse=True)
                cnn_output = tf.concat([cnn_ans, cnn_ques], -1)
            hidden=self.lstm_module(cnn_output)
        else:
            raise ValueError('clf should be one of [cnn, lstm, clstm]')
        logits,predictions=self.softmax_module(hidden)
        return logits,predictions

    def model_fn(self,features,labels,mode):
        logits, predictions=self.forward(features,mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss, train_op = self.loss_module(logits, labels)

        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predictions,
                                       name='acc_op')
        confusion_matrix=eval_confusion_matrix(labels,predictions,args.num_class)

        metrics = {'accuracy': accuracy,"confusion_matrix":confusion_matrix}
        tf.summary.scalar('accuracy', accuracy[1])

        if not args.pretrain:
            exclude = ['pt_softmax/']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(args.model_path,
                                          {v.name.split(':')[0]: v for v in variables_to_restore})
            #print(variables_to_restore)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        if mode == tf.estimator.ModeKeys.TRAIN:
           logging_hook = tf.train.LoggingTensorHook({'global_step': self.global_step,
                                                      "lr":self.learning_rate_exp}, every_n_iter=100)
           return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


    def softmax_module(self,input,reuse=False):
        if not args.pretrain:
            with tf.variable_scope("ft_softmax", reuse=reuse,regularizer=tf.contrib.layers.l2_regularizer(args.l2)):
                output=tf.layers.dense(input,args.num_class)
        else :
            with tf.variable_scope("pt_softmax", reuse=reuse,regularizer=tf.contrib.layers.l2_regularizer(args.l2)):
                output=tf.layers.dense(input,self.vocab_size)

        logits=tf.nn.softmax(output)
        predictions=tf.argmax(logits,-1)
        return logits,predictions

    def loss_module(self,logits,labels):
        self.global_step = tf.train.get_global_step()
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_losses
        self.learning_rate_exp = tf.train.exponential_decay(args.lr, self.global_step,
                                                            args.decay_size,
                                                            args.decay_factor)
        params = tf.trainable_variables()
        optmizer = tf.train.AdamOptimizer(self.learning_rate_exp)
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.grad_clip)
        gard_op = optmizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        var_avg = tf.train.ExponentialMovingAverage(args.moving_average_decay,self.global_step)
        variables_averages_op = var_avg.apply(params)
        with tf.control_dependencies([gard_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return loss,train_op

    def get_seq_len(self,ans,ques):
        ans_len=tf.count_nonzero(ans,-1)
        ques_len=tf.count_nonzero(ques,-1)
        cmp=tf.greater(ans_len,ques_len)
        seq_len=tf.where(cmp,ans_len,ques_len)
        return seq_len