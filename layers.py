import tensorflow.compat.v1 as tf
import helper as H

helper = H.helper(tf)

def conv(input, reuse=False, name='conv'):
  with tf.variable_scope(name, reuse=reuse):
    weights = helper.genWeights("weights",shape=[7, 7, input.get_shape()[3], 3])
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding = 'SAME')
    print('first conv '+str(conv.shape))
    return tf.nn.tanh(helper.normalize(conv))

def genr(input, k, reuse=False, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights = helper.genWeights("weights",shape=[3, 3, input.get_shape()[3], k])
    conv = tf.nn.conv2d(input, weights, strides=[1, 2, 2, 1], padding='SAME')
    print('conv shape : {}'+ str(conv.shape))

    return tf.nn.relu(helper.normalize(conv))

def addResBlocks(input, reuse):
  output = res(input, input.get_shape()[3], reuse)

  for i in range(1,10):
    output = res(output, output.get_shape()[3], reuse)

  return output

def res(input, n, reuse=False):
  name = 'RESBLOCK{}{}'.format(input.get_shape()[3], 1)

  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('resLayer', reuse=reuse):

      resWeights = helper.genWeights("resWeights", shape=[3, 3, input.get_shape()[3], n])
      conv = tf.nn.conv2d(input, resWeights, strides=[1, 1, 1, 1], padding='SAME')
      relued = tf.nn.relu(helper.normalize(conv))

    with tf.variable_scope('resLayer2', reuse=reuse):
      resWeights2 = helper.genWeights("resWeights2",shape=[3, 3, relued.get_shape()[3], n])

    return input+helper.normalize(tf.nn.conv2d(relued, resWeights2, strides=[1, 1, 1, 1], padding='SAME'))



def frac(input, k, reuse=False, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights = helper.genWeights("weights", shape=[3, 3, k, input.get_shape().as_list()[3]])
    outputShape = [input.get_shape().as_list()[0], input.get_shape().as_list()[1]*2, input.get_shape().as_list()[1]*2, k]
    deConvConv = tf.nn.conv2d_transpose(input, weights, output_shape=outputShape, strides=[1, 2, 2, 1], padding='SAME')
    print('deconv shape : {}'+ str(deConvConv.shape))
    return tf.nn.relu(helper.normalize(deConvConv))

def disc(input, k, reuse=False, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights = helper.genWeights("weights", shape=[4, 4, input.get_shape()[3], k])
    conv = tf.nn.conv2d(input, weights, strides=[1, 2, 2, 1], padding='SAME')

    return helper.lRelu(helper.normalize(conv), 0.3)

def disc2(input, k, reuse=False, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights = helper.genWeights("weights", shape=[4, 4, input.get_shape()[3], k])
    conv = tf.nn.conv2d(input, weights, strides=[1, 2, 2, 1], padding='SAME')

    return tf.maximum(tf.contrib.layers.batch_norm(input, decay=0.9,scale=True,updates_collections=None,is_training=True), 0)


def decide(input, reuse=False, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights = helper.genWeights("weights", shape=[4, 4, input.get_shape()[3], 1])
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    biases = helper.genBiases("biases", [1])

    return conv + biases