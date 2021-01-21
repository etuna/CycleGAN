class helper:
    def __init__(self, tf):
        self.tf = tf

    def genWeights(self, name, shape, mean=0.0, stddev=0.03):
        return self.tf.get_variable(name, shape,initializer=self.tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=self.tf.float32))

    def genBiases(self, name, shape):
        return self.tf.get_variable(name, shape, initializer=self.tf.constant_initializer(0))

    def normalize(self, input):
        return self.instanceNormalize(input)

    def sLog(self, var, eps=1e-13):
        return self.tf.log(var + eps)

    def relu(self, input):
        return self.tf.maximum(0, input)

    def lRelu(self, input, slope):
        return self.tf.maximum(slope * input, input)

    def instanceNormalize(self, input):
        with self.tf.variable_scope("instanceNorm"):
            depth = input.get_shape()[3]
            biases = self.genBiases("offset", [depth])
            weights = self.genWeights("scale", [depth], mean=1.0)
            mean, variance = self.tf.nn.moments(input, axes=[1, 2], keep_dims=True)

            return weights * ((input - mean) * self.tf.rsqrt(variance + 1e-5)) + biases
