import numpy as np
import tensorflow as tf
import pdb
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

def write_tfrecord(features, scene_images, labels, goal_snpts,file_location,meta_data_dict):

  writer = tf.io.TFRecordWriter(file_location)

  for feature, scene_image, label,  goal_snpt in zip(features,scene_images,labels,goal_snpts):

      ftr = {'image_hist'   : _bytes_feature_list(tf.compat.as_bytes(scene_image.tostring())),
             'image_size'   : _bytes_feature_list(np.array(scene_image.shape,np.int32).tostring()),
             'feature'      : _bytes_feature_list(tf.io.serialize_tensor(feature)),
             'feature_size' : _bytes_feature_list(np.array(feature.shape,np.int32).tostring()),
             'label'        : _bytes_feature_list(tf.io.serialize_tensor(label)),
             'label_size'   : _bytes_feature_list(np.array(label.shape,np.int32).tostring()),
             'goal_snpt'    : _bytes_feature_list(tf.io.serialize_tensor(goal_snpt)),
             'goal_snpt_size' : _bytes_feature_list(np.array(goal_snpt.shape,np.int32).tostring()),
            }
      #pdb.set_trace()
      example = tf.train.Example(features = tf.train.Features(feature=ftr))
      writer.write(example.SerializeToString())

  writer.close()
	#pass

def _parse_function(proto):
    ftr = {'image_hist'     : tf.io.FixedLenFeature([], tf.string),
           'image_size'     : tf.io.FixedLenFeature([], tf.string),
           'feature'        : tf.io.FixedLenFeature([], tf.string),
           'feature_size'   : tf.io.FixedLenFeature([], tf.string),
           'label'          : tf.io.FixedLenFeature([], tf.string),
           'label_size'     : tf.io.FixedLenFeature([], tf.string),
           'goal_snpt'      : tf.io.FixedLenFeature([], tf.string),
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

def read_tfrecord(files):

  dataset = tf.data.TFRecordDataset(files)
  dataset = dataset.map(_parse_function)

  dataset = dataset.shuffle(40)

  iterator = dataset.__iter__()


  image, feature, label, goal = iterator.get_next()

  return image, feature, label, goal







#image = np.zeros((100,50),dtype=np.uint8)
#image[0,:] =  1
#image[10,:] = 1
#hist = np.random.randn(19)
#print(len(hist))
#label = 1

#writer = tf.io.TFRecordWriter('tfrecord.tf')

#feature = {'image' : _bytes_feature_list(tf.compat.as_bytes(image.tostring())),
#           '
#           'label' : _int64_feature(int(label)),
#           'hist' : _float_feature(hist)}
