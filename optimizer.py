def optimizer(tf, G, D_Y, F, D_X, learningRate, GLoss, dyLoss, FLoss, dxLoss):
    def opt(Gloss, variables):
        learning_rate = (tf.where(tf.greater_equal(tf.Variable(0, trainable=False), 1000),
                                  tf.train.polynomial_decay(learningRate, tf.Variable(0, trainable=False) - 1000,
                                                            1000, 0, power=1.0), learningRate))
        return (tf.train.AdamOptimizer(learning_rate, beta1=0.5, name='Adam').minimize(Gloss, global_step=tf.Variable(0,
                                                                                                                      trainable=False),
                                                                                       var_list=variables))

    with tf.control_dependencies(
            [opt(GLoss, G.trainVars), opt(dyLoss, D_Y.trainVars), opt(FLoss, F.trainVars),
             opt(dxLoss, D_X.trainVars)]):
        return tf.no_op(name='optimizers')
