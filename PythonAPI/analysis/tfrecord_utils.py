import numpy as np
import tensorflow as tf

# Helper functions to prepare and parse a TFRecord-based dataset.  Used in conjunction with 
# snippet generation code in pkl_reader.py.

# Standard TFRecord feature generation.  See https://www.tensorflow.org/tutorials/load_data/tfrecord
def _int64_feature_list(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature_list(value):
  if isinstance(value,type(tf.constant(0))):
    value = value.numpy()

  if not isinstance(value, list):
    value = [value]

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature_list(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))
######################################################################################################
# TFRecord writing for our dataset.
def write_tfrecord(features, scene_images, labels, goal_snpts,file_location,meta_data_dict):
  # Note: meta_data_dict is not used.  Can be incorporated into the tfrecord in the future.
  writer = tf.io.TFRecordWriter(file_location)
  
  for feature, scene_image, label,  goal_snpt in zip(features,scene_images,labels,goal_snpts):

      ftr = {'image_hist'   : _bytes_feature_list(tf.compat.as_bytes(scene_image.tostring())),           # image history for LSTM
             'image_size'   : _bytes_feature_list(np.array(scene_image.shape,np.int32).tostring()),
             'feature'      : _bytes_feature_list(tf.io.serialize_tensor(feature)),                      # motion history for LSTM    
             'feature_size' : _bytes_feature_list(np.array(feature.shape,np.int32).tostring()),
             'label'        : _bytes_feature_list(tf.io.serialize_tensor(label)),                        # future motion and intent label
             'label_size'   : _bytes_feature_list(np.array(label.shape,np.int32).tostring()),
             'goal_snpt'    : _bytes_feature_list(tf.io.serialize_tensor(goal_snpt)),                    # occupancy information
             'goal_snpt_size' : _bytes_feature_list(np.array(goal_snpt.shape,np.int32).tostring()),
            }
      example = tf.train.Example(features = tf.train.Features(feature=ftr))
      writer.write(example.SerializeToString())

  writer.close()

# Main parsing function used for the TFRecord.
# Used in individual model files for parsing and reading.  Refer to them for usage details.
def _parse_function(proto):
    ftr = {'image_hist'     : tf.io.FixedLenFeature([], tf.string), # image history for LSTM
           'image_size'     : tf.io.FixedLenFeature([], tf.string), 
           'feature'        : tf.io.FixedLenFeature([], tf.string), # motion history for LSTM
           'feature_size'   : tf.io.FixedLenFeature([], tf.string),
           'label'          : tf.io.FixedLenFeature([], tf.string), # future motion and intent label
           'label_size'     : tf.io.FixedLenFeature([], tf.string),
           'goal_snpt'      : tf.io.FixedLenFeature([], tf.string), # occupancy information
           'goal_snpt_size' : tf.io.FixedLenFeature([], tf.string),
          }
    parsed_features = tf.io.parse_single_example(proto,ftr)

    # Parse image and image size
    parsed_features['image_hist'] = tf.io.decode_raw(parsed_features['image_hist'],tf.uint8)
    parsed_features['image_size'] = tf.io.decode_raw(parsed_features['image_size'],tf.int32)

    image_size = parsed_features['image_size']
    image = tf.reshape(parsed_features['image_hist'],image_size)

    # Parse features
    feature_size = tf.io.decode_raw(parsed_features['feature_size'],tf.int32)
    feature = tf.reshape(tf.io.parse_tensor(parsed_features['feature'],out_type=tf.float64),feature_size)

    # Label features
    label_size = tf.io.decode_raw(parsed_features['label_size'],tf.int32)
    label = tf.reshape(tf.io.parse_tensor(parsed_features['label'],out_type=tf.float64),label_size)

    # Label features
    goal_size = tf.io.decode_raw(parsed_features['goal_snpt_size'],tf.int32)
    goal = tf.reshape(tf.io.parse_tensor(parsed_features['goal_snpt'],out_type=tf.float64),goal_size)

    return image, feature, label, goal