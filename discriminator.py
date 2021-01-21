import tensorflow.compat.v1 as tf
import layers

class Discriminator:
  def __init__(self, name):
    self.name = name
    self.reuse = False

  def __call__(self, input):
    with tf.variable_scope(self.name):

      # First Disc
      disc1 = self.getDisc(input)
      output1 = layers.decide(disc1[3], reuse=self.reuse, name='output1')

      # Second Disc
      disc2 = self.getDisc(input)
      output2 = layers.decide(disc2[3], reuse=self.reuse, name='output2')

    self.trainVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return tf.math.minimum(output1, output2)

  def getDisc(self, input):
    d64 = layers.disc(input, 64, reuse=self.reuse, name='d64')
    d128 = layers.disc(d64, 128, reuse=self.reuse, name='d128')
    d256 = layers.disc(d128, 256, reuse=self.reuse, name='d256')
    d512 = layers.disc(d256, 512, reuse=self.reuse, name='d512')
    if not self.reuse:
      self.reuse = tf.AUTO_REUSE
    return d64, d128, d256, d512