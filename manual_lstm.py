import tensorflow as tf

class ManualLSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ManualLSTM, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', trainable=True)
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', trainable=True)
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', trainable=True)
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.U_o = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', trainable=True)
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        h = tf.zeros((batch_size, self.units))
        c = tf.zeros((batch_size, self.units))

        for t in range(inputs.shape[1]):
            x_t = inputs[:, t, :]

            f = tf.sigmoid(tf.matmul(x_t, self.W_f) + tf.matmul(h, self.U_f) + self.b_f)
            i = tf.sigmoid(tf.matmul(x_t, self.W_i) + tf.matmul(h, self.U_i) + self.b_i)
            o = tf.sigmoid(tf.matmul(x_t, self.W_o) + tf.matmul(h, self.U_o) + self.b_o)
            c_tilde = tf.tanh(tf.matmul(x_t, self.W_c) + tf.matmul(h, self.U_c) + self.b_c)

            c = f * c + i * c_tilde
            h = o * tf.tanh(c)

        return h
