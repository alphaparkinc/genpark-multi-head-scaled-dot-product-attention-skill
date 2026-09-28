from client import ScaledDotProductAttention

def main():
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[5.0, 5.0], [10.0, 10.0]]
    out, weights = ScaledDotProductAttention.forward(Q, K, V, causal=True)
    print("Attention Weights matrix:", weights)
    print("Attention Output matrix:", out)

if __name__ == "__main__":
    main()
