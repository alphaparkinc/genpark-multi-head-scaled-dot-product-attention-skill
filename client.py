"""Transformer Scaled Dot-Product Self-Attention Engine.
100% Python Standard Library.
"""

import math

class ScaledDotProductAttention:
    """Scaled dot-product attention Softmax(Q*K^T / sqrt(d_k)) * V with optional causal masking."""

    @staticmethod
    def forward(Q: list, K: list, V: list, causal: bool = False) -> tuple:
        seq_len = len(Q)
        d_k = len(Q[0])
        scale = 1.0 / math.sqrt(d_k)

        scores = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                if causal and j > i:
                    row.append(-1e9)
                else:
                    dot = sum(Q[i][k] * K[j][k] for k in range(d_k)) * scale
                    row.append(dot)
            scores.append(row)

        weights = []
        for row in scores:
            max_v = max(row)
            exps = [math.exp(v - max_v) for v in row]
            sum_exp = sum(exps)
            weights.append([e / sum_exp for e in exps])

        d_v = len(V[0])
        out = []
        for i in range(seq_len):
            out_row = []
            for k in range(d_v):
                val = sum(weights[i][j] * V[j][k] for j in range(seq_len))
                out_row.append(val)
            out.append(out_row)
        return out, weights
