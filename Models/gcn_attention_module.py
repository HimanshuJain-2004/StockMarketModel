import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import MinMaxNorm
from temporal_attention import TemporalAttentionBlock


class GCNTemporalAttention(tf.keras.Model):
    """
    REGCN-faithful GCN + Temporal Self-Attention (GRU replaced)
    """

    def __init__(self, Madj, n_nodes, hidden_dim, s_index,
                 num_heads=4, dropout=0.2,**kwargs):
        super().__init__(**kwargs)

        """
        Madj: (K, N, N) Laplacian-normalized adjacency matrices
        """

        self.Madj = Madj
        self.K = Madj.shape[0]
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.s_index = s_index
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.wa = self.add_weight(
            shape=(self.K,),
            initializer='random_normal',
            trainable=True,
            constraint=MinMaxNorm(0.0, 1.0),
            name="wa"
        )
        self.w_gcn = self.add_weight(
            shape=(self.n_nodes, self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="w_gcn"
        )

        self.temporal_attn = TemporalAttentionBlock(
            d_model=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.out_dense = Dense(1)
        
    def get_config(self):
        import numpy as np
        config = super().get_config()

        if isinstance(self.Madj, tf.Tensor):
            Madj_list = self.Madj.numpy().tolist()
        else:
            Madj_list = np.array(self.Madj).tolist()

        config.update({
            "Madj": Madj_list,
            "n_nodes": self.n_nodes,
            "hidden_dim": self.hidden_dim,
            "s_index": self.s_index,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        import numpy as np
        Madj = tf.convert_to_tensor(np.array(config["Madj"], dtype=np.float32))
        config["Madj"] = Madj
        return cls(**config)

    def call(self, x, training=False):
        """
        x: (B, T, N)
        returns: (B, T, 1)
        """
        A = tf.reduce_sum(
            self.wa[:, None, None] * self.Madj,
            axis=0
        )  # (N, N)

        Ax = tf.linalg.matmul(x, A)
        H = tf.tensordot(Ax, self.w_gcn, axes=[[2], [0]])
        H = tf.nn.relu(H)

        H_attn = self.temporal_attn(H, training=training)

        y_pred = self.out_dense(H_attn)  

        return y_pred