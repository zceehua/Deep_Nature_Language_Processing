import tensorflow as tf
from config import args
import os

PAD_INDEX=0
class Parser(object):
    @staticmethod
    def _parse_train_fn(serialized_example):
        parsed_features = tf.parse_single_example(
            serialized_example,
            features={
                'answer': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True,default_value=PAD_INDEX),
                'question': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True,default_value=PAD_INDEX),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        return parsed_features["answer"],parsed_features["question"],parsed_features["label"]

    @staticmethod
    def _parse_test_fn(serialized_example):
        parsed_features = tf.parse_single_example(
            serialized_example,
            features={
                'answer': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True,default_value=PAD_INDEX),
                'question': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True,default_value=PAD_INDEX),
            })
        return parsed_features["answer"],parsed_features["question"]

class RecordLoader(object):

    def train_input(self,name,batch_size=args.batch_size):
        path=os.path.join(args.save_path,name)
        data=tf.data.TFRecordDataset([path])
        data=data.map(Parser._parse_train_fn,num_parallel_calls=4)
        #if using FixedLenFeature,then shape should be set to [], as the input len of
        #FixedLenFeature is fixed
        data=data.padded_batch(batch_size,padded_shapes=([args.max_len],[args.max_len],[]))
        #print(data)
        #data=data.filter(lambda x, y, z: tf.equal(tf.shape(x), batch_size))
        iterator = data.make_one_shot_iterator()
        ans, ques, y = iterator.get_next()
        return ({'answer': ans, 'question': ques}, y)

    def test_input(self,name="test"):
        path=os.path.join(args.save_path,"{}.tfrecord".format(name))
        data=tf.data.TFRecordDataset(path)
        data.map(Parser._parse_test_fn,num_parallel_calls=4)
        data = data.padded_batch(args.batch_size, padded_shapes=([args.max_len],[args.max_len]))
        iterator = data.make_one_shot_iterator()
        ans, ques = iterator.get_next()
        return ({'answer': ans, 'question': ques})