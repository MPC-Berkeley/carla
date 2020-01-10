import numpy as np 
import tensorflow as tf 
impo

def _int64_feature_list(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature_list(value):	
	if not isinstance(value, list):
		value = [value]	
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature_list(value):
	if not isinstance(value, list):
		value = [value]	
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_tfrecord():
	pass

