import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from utils import calculate_laplacian
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, tanh
# from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, GRUCell, CuDNNLSTM, BatchNormalization, RNN, TimeDistributed
from tensorflow.keras.constraints import MinMaxNorm, UnitNorm

class gcgru(Layer):

    # def __init__(self, num_units, adj1,adj2, num_gcn_nodes, s_index, **kwargs):
    def __init__(self, num_units, adj, num_gcn_nodes, s_index,n_graphs=3, **kwargs):
        super(gcgru, self).__init__(**kwargs)
        self.units = num_units
        self._gcn_nodes = num_gcn_nodes
        self.s_index = s_index
        self.n_graphs = n_graphs
        self._adj = adj

    @ property
    def state_size(self):
        return self.units

    def get_config(self):
        config = super().get_config()

        if isinstance(self._adj, tf.Tensor):
            adj_list = self._adj.numpy().tolist()
        else:
            adj_list = np.array(self._adj).tolist()

        config.update({
            "num_units": self.units,
            "adj": adj_list,
            "num_gcn_nodes": self._gcn_nodes,
            "s_index": self.s_index,
            "n_graphs": self.n_graphs,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # adj was saved as list → convert back to np.array
        import numpy as np
        config["adj"] = np.array(config["adj"], dtype=np.float32)
        return cls(**config)
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        # weights
        self.wz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wz')
        # self.wr = self.add_weight(shape=(self.units, self.units),
        #                           initializer='random_normal',
        #                           trainable=True,
        #                           name='wr')
        self.wh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='wh')

        # self.w0 = self.add_weight(shape=(1, self.units),
        #                           initializer='random_normal',
        #                           trainable=True,
        #                           name='w0')
        
        self.w_gcn = self.add_weight(
                                    shape=(self._gcn_nodes, self.units),
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    name='w_gcn'
                                )

        # self.wa = self.add_weight(shape=(self.n_graphs,self._gcn_nodes,self._gcn_nodes),
        #                           initializer='random_normal',
        #                           trainable=True,
        #                           constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
        #                           # constraint= UnitNorm(axis=0),
        #                           name='wa')
        self.wa = self.add_weight(
                                shape=(self._adj.shape[0],),
                                initializer='random_normal',
                                trainable=True,
                                constraint=MinMaxNorm(0.0, 1.0),
                                name='wa'
                            )
        # self.adj_max = tf.reduce_sum(
        #     self.wa[:, None, None] * self._adj,
        #     axis=0
        # )

        # us
        self.uz = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='uz')
        # self.ur = self.add_weight(shape=(self.units, self.units),
        #                           initializer='random_normal',
        #                           trainable=True,
        #                           name='ur')
        self.uh = self.add_weight(shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='uh')
        # biases
        self.bz = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bz")
        # self.br = self.add_weight(
        #     shape=(self.units,), initializer="random_normal", trainable=True, name="br")
        self.bh = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True, name="bh")
        self.built = True

    def call(self, inputs, states):
        state = states[0]
        adj_max = tf.reduce_sum(
            self.wa[:, None, None] * self._adj,
            axis=0
        )
        # adj_max = self.adj_max
        #GCN
        x = self.gc(inputs, adj_max)
        #GRU
        z = sigmoid(
            K.dot(x, self.wz) + K.dot(state, self.uz) + self.bz
        )

        # r = sigmoid(
        #     K.dot(x, self.wr) + K.dot(state, self.ur) + self.br
        # )

        # h_tilde = tanh(
        #     K.dot(x, self.wh) + K.dot(r * state, self.uh) + self.bh
        # )
        
        h_tilde = tanh(
            K.dot(x, self.wh) + K.dot(state, self.uh) + self.bh
        )

        output = (1 - z) * state + z * h_tilde
        # output = z * state + (1 - z) * h
        return output, output

    def gc(self, inputs, adj):
        # adj is already a normalized Laplacian (Tensor)
        ax = tf.matmul(inputs,adj)     # (N, N) × (batch, N)
        # ax = ax[:, self.s_index]        # select target node
        x = tf.matmul(ax, self.w_gcn)
        x = tf.nn.relu(x)
        return x



