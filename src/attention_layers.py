"""Mecanismos de atención y bloques Transformer implementados desde cero en Keras 3 (backend TensorFlow).

Referencia: Vaswani et al. (2017), "Attention Is All You Need".
"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops


# ============================================================================ atención básica
def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k + máscara) · V      (ecuación 1 del paper)

    q: (..., Tq, d_k)  k: (..., Tk, d_k)  v: (..., Tk, d_v)
    mask: 1 = posición permitida, 0 = bloqueada; se suma -1e9 a las bloqueadas antes del softmax.
    Devuelve (salida (..., Tq, d_v), pesos de atención (..., Tq, Tk)).
    """
    d_k = ops.cast(ops.shape(k)[-1], q.dtype)
    scores = ops.matmul(q, ops.swapaxes(k, -1, -2)) / ops.sqrt(d_k)
    if mask is not None:
        scores = scores + (1.0 - ops.cast(mask, scores.dtype)) * -1e9
    weights = ops.softmax(scores, axis=-1)
    return ops.matmul(weights, v), weights


def causal_mask(T):
    """Máscara triangular inferior: la posición t sólo puede atender a posiciones ≤ t."""
    return np.tril(np.ones((T, T), dtype="float32"))


@keras.saving.register_keras_serializable(package="tst")
class AdditiveAttention(layers.Layer):
    """Atención aditiva de Bahdanau sobre los estados ocultos de un codificador recurrente.

    e_t = vᵀ tanh(W₁ h_t + W₂ s),  α = softmax(e),  c = Σ_t α_t h_t
    donde h_t son los estados del LSTM en cada día y s es el último estado (la "consulta").
    """
    def __init__(self, units=32, **kw):
        super().__init__(**kw); self.units = units
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=True)
        self.v = layers.Dense(1, use_bias=False)

    def call(self, states, query):
        score = self.v(ops.tanh(self.W1(states) + ops.expand_dims(self.W2(query), 1)))   # (B, T, 1)
        alpha = ops.softmax(score, axis=1)
        context = ops.sum(alpha * states, axis=1)                                         # (B, units_lstm)
        return context, ops.squeeze(alpha, -1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# ============================================================================ Transformer
@keras.saving.register_keras_serializable(package="tst")
class PositionalEncoding(layers.Layer):
    """Codificación posicional sinusoidal (sección 3.5 del paper):
    PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).
    Se SUMA al embedding: la atención por sí sola no conoce el orden temporal."""
    def __init__(self, max_len=512, d_model=32, **kw):
        super().__init__(**kw); self.max_len, self.d_model = max_len, d_model
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
        pe = np.zeros((max_len, d_model), dtype="float32")
        pe[:, 0::2], pe[:, 1::2] = np.sin(angle[:, 0::2]), np.cos(angle[:, 1::2])
        self.pe = pe

    def call(self, x):
        T = ops.shape(x)[1]
        return x + ops.convert_to_tensor(self.pe)[None, :T, :]

    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len, "d_model": self.d_model}


@keras.saving.register_keras_serializable(package="tst")
class MultiHeadAttentionScratch(layers.Layer):
    """Multi-head attention (sección 3.2.2):
    MultiHead(Q,K,V) = Concat(head₁..head_h)·Wᴼ,   head_i = Attention(Q·Wᵢ^Q, K·Wᵢ^K, V·Wᵢ^V).
    Las h cabezas trabajan en paralelo en subespacios de dimensión d_k = d_model / h."""
    def __init__(self, d_model=32, num_heads=4, **kw):
        super().__init__(**kw)
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        self.d_model, self.h, self.d_k = d_model, num_heads, d_model // num_heads
        self.wq, self.wk, self.wv = (layers.Dense(d_model, name=n) for n in ("wq", "wk", "wv"))
        self.wo = layers.Dense(d_model, name="wo")

    def _split(self, x):                       # (B, T, d_model) -> (B, h, T, d_k)
        B, T = ops.shape(x)[0], ops.shape(x)[1]
        return ops.transpose(ops.reshape(x, (B, T, self.h, self.d_k)), (0, 2, 1, 3))

    def call(self, query, value, key=None, mask=None, return_attention_scores=False):
        key = value if key is None else key
        q, k, v = self._split(self.wq(query)), self._split(self.wk(key)), self._split(self.wv(value))
        out, w = scaled_dot_product_attention(q, k, v, mask)           # (B, h, Tq, d_k), (B, h, Tq, Tk)
        B, Tq = ops.shape(out)[0], ops.shape(out)[2]
        out = self.wo(ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, Tq, self.d_model)))
        return (out, w) if return_attention_scores else out

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "num_heads": self.h}


@keras.saving.register_keras_serializable(package="tst")
class EncoderBlock(layers.Layer):
    """Capa del codificador: self-attention multi-cabeza + Add & Norm + red feed-forward + Add & Norm."""
    def __init__(self, d_model=32, num_heads=4, d_ff=64, dropout=0.1, **kw):
        super().__init__(**kw)
        self.cfg = dict(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.mha = MultiHeadAttentionScratch(d_model, num_heads)
        self.ffn = keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.norm1, self.norm2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.drop1, self.drop2 = layers.Dropout(dropout), layers.Dropout(dropout)

    def call(self, x, training=False, mask=None, return_attention=False):
        a, w = self.mha(x, x, mask=mask, return_attention_scores=True)
        x = self.norm1(x + self.drop1(a, training=training))                 # residual + LayerNorm
        x = self.norm2(x + self.drop2(self.ffn(x), training=training))
        return (x, w) if return_attention else x

    def get_config(self):
        return {**super().get_config(), **self.cfg}


@keras.saving.register_keras_serializable(package="tst")
class DecoderBlock(layers.Layer):
    """Capa del decodificador: self-attention ENMASCARADA (causal) + atención cruzada al codificador + FFN."""
    def __init__(self, d_model=32, num_heads=4, d_ff=64, dropout=0.1, **kw):
        super().__init__(**kw)
        self.cfg = dict(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.self_mha = MultiHeadAttentionScratch(d_model, num_heads)
        self.cross_mha = MultiHeadAttentionScratch(d_model, num_heads)
        self.ffn = keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.n1, self.n2, self.n3 = (layers.LayerNormalization(epsilon=1e-6) for _ in range(3))
        self.d1, self.d2, self.d3 = (layers.Dropout(dropout) for _ in range(3))

    def call(self, y, enc, training=False, return_attention=False):
        T = ops.shape(y)[1]
        mask = ops.tril(ops.ones((T, T)))                                  # causal: no mira el futuro
        a, w_self = self.self_mha(y, y, mask=mask, return_attention_scores=True)
        y = self.n1(y + self.d1(a, training=training))
        c, w_cross = self.cross_mha(y, enc, return_attention_scores=True)  # Q del decoder, K y V del encoder
        y = self.n2(y + self.d2(c, training=training))
        y = self.n3(y + self.d3(self.ffn(y), training=training))
        return (y, w_self, w_cross) if return_attention else y

    def get_config(self):
        return {**super().get_config(), **self.cfg}
