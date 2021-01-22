import tensorflow.compat.v1 as tf
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
import optimizer as opter

class CycleGAN:
  def __init__(self,
               X_train_file='',
               Y_train_file=''
              ):

    self.X_train_file = X_train_file
    self.Y_train_file = Y_train_file
    self.learningRate = 2e-4

    self.G = Generator('Generator')
    self.D_Y = Discriminator('DiscriminatorY')
    self.F = Generator('Fraction')
    self.D_X = Discriminator('DiscriminatorX')

    X_reader = Reader(self.X_train_file, name='X', image_size=256, batch_size=1)
    Y_reader = Reader(self.Y_train_file, name='Y', image_size=256, batch_size=1)

    self.Xs = X_reader.feed()
    self.Ys = Y_reader.feed()

  def model(self):

    x = self.Xs
    y = self.Ys

    self.fake_x = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])
    self.fake_y = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])

    cycle_loss = self.cyclicLoss(self.G, self.F, x, y)

    # Generating the fake image of y, X -> Fake Y
    fake_y = self.G(x)
    GLoss = self.generatorLoss(self.D_Y, fake_y)+ cycle_loss
    dyLoss = self.discriminatorLoss(self.D_Y, y, self.fake_y)

    # Reconstruction - Y -> Fake X
    fake_x = self.F(y)
    FLoss = self.generatorLoss(self.D_X, fake_x) + cycle_loss
    dxLoss = self.discriminatorLoss(self.D_X, x, self.fake_x)

    return GLoss, dyLoss, FLoss, dxLoss, fake_y, fake_x

  def discriminatorLoss(self, D, y, fake_y):

    realLoss = tf.reduce_mean(tf.squared_difference(D(y), 0.9))
    fakeLoss = tf.reduce_mean(tf.square(D(fake_y)))

    return (realLoss + fakeLoss) / 2

  def generatorLoss(self, D, fake_y):
    return tf.reduce_mean(tf.squared_difference(D(fake_y), 0.9))

  def cyclicLoss(self, G, F, x, y):

    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))

    return 10*forward_loss + 10*backward_loss # lambda = 10

  def optimizer(self, GLoss, dyLoss, FLoss, dxLoss):
    return opter.optimizer(tf, self.G, self.D_Y, self.F, self.D_X, learningRate=self.learningRate, GLoss=GLoss, dyLoss=dyLoss, FLoss=FLoss, dxLoss=dxLoss)
