# Cross Entropy
#cross-entropy measures **how well a set of estimated probabilities matches the true classes** — equivalently, the **average number of bits you send per option** when you encode events using an assumed distribution instead of the true one. It originates from Claude Shannon's information theory.

Suppose you want to transmit the daily weather, and there are 8 equally likely options (sunny, rainy, etc.). You could encode each one with 3 bits, since $2^3 = 8$. But if you expect it to be sunny *almost* every day, it's more efficient to encode "sunny" on a single bit (0) and the other seven options on longer codes (starting with a 1). Cross entropy measures the average number of bits you actually end up sending.

- If your assumed distribution is **perfect**, cross entropy equals the **entropy** of the source itself — its intrinsic unpredictability.
- If your assumption is **wrong** (e.g. it rains often), cross entropy is **larger**, by an amount called the **Kullback–Leibler (KL) divergence**.

For two discrete probability distributions $p$ (true) and $q$ (estimated), it is defined as:
$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

Cross entropy is widely used as a **classification cost function**, because it penalizes the model for assigning a low probability to the true class. For example, see [[Softmax Regression|softmax regression]].