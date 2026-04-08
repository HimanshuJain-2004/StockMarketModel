import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention

class TemporalAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=4, dropout=0.2, **kwargs):
        super().__init__(**kwargs)

        # Save init args for serialization
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            Dense(d_model * 4, activation='relu'),
            Dense(d_model)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)

        self.gate = Dense(d_model, activation='sigmoid')

    def call(self, x, training=False):
        attn_out = self.mha(x, x, x, training=training)
        x1 = self.norm1(x + attn_out)

        ffn_out = self.ffn(x1)
        ffn_out = self.dropout(ffn_out, training=training)
        x2 = self.norm2(x1 + ffn_out)

        G = self.gate(x2)
        return G * x2 + (1 - G) * x

    # ==================================================
    #  🔥 SERIALIZATION SUPPORT
    # ==================================================
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)