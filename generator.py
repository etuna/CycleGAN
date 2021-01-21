import tensorflow.compat.v1 as tf

import layers
import utils

class Generator:
  def __init__(self, name):
    self.name = name
    self.reuse = False

  def __call__(self, input):
    with tf.variable_scope(self.name):
      # Generator
      genr = self.getConvLayers(input)

      # Deconv
      deConv = self.deConv(self.getResOutput(genr[2]))

    self.trainVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return layers.conv(deConv[0], reuse=self.reuse, name='output')

  def sample(self, input):
    return tf.image.encode_jpeg(tf.squeeze(utils.batch_convert2int(self.__call__(input)), [0]))

  def getConvLayers(self, input):
    conv = layers.conv(input, reuse=self.reuse, name='conv')
    genr64 = layers.genr(conv, 128, reuse=self.reuse, name='genr64')
    genr128 = layers.genr(genr64, 256, reuse=self.reuse, name='genr128')

    if not self.reuse:
      self.reuse = tf.AUTO_REUSE

    return conv, genr64, genr128

  def getResOutput(self, genr):
    return layers.addResBlocks(genr, reuse=self.reuse)

  def deConv(self, resOutput):
    frac64 = layers.frac(resOutput, 128, reuse=self.reuse, name='frac64')
    frac32 = layers.frac(frac64, 64,  reuse=self.reuse, name='frac32')
    return frac32, frac64
