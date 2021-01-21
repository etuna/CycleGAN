import tensorflow.compat.v1 as tf

def getTfConfig(tf):
    tf.app.flags.DEFINE_string('X', 'data/tfrecords/apple.tfrecords','')
    tf.app.flags.DEFINE_string('Y', 'data/tfrecords/orange.tfrecords','')
    return tf