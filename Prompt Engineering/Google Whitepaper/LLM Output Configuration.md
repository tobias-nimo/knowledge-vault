Most LLMs come with various **configuration options that control the LLM’s output**. Effective prompt engineering requires setting these configurations optimally for your task.
## Output length 📏
An important configuration setting is the **number of tokens to generate in a response**.

Reducing the output length of the LLM doesn’t cause the LLM to become more stylistically or textually succinct in the output it creates, it just causes the LLM to stop predicting more tokens once the limit is reached.

Be aware, generating more tokens requires more computation from the LLM, leading to higher energy consumption and potentially slower response times, which leads to higher costs.
## Sampling controls 🎚️
LLMs do not formally predict a single token. Rather, LLMs predict probabilities for what the next token could be, with each token in the LLM’s vocabulary getting a probability. Those **token probabilities are then sampled** to determine what the next produced token will be.

This are the most common configuration settings that determine how predicted token probabilities are processed to choose a single output token:
- **Temperature** 🌡️controls randomness in token selection—lower values yield more predictable and factual outputs, while higher values increase diversity and unpredictability. **Greedy decoding** 🥶: when temperature is set to 0 the highest probability token is always selected (deterministic token selection).
- **Top-K sampling** limits next-token choices to the top K most probable options, balancing creativity (higher K) and factuality (lower K); K=1 equals greedy decoding.
- **Top-P sampling** selects tokens until their cumulative probability reaches a threshold P (0 to 1).

**Putting it all together**: tokens that meet both the top-K and top-P criteria are candidates for the next predicted token, and then temperature is applied to sample from them. 

**Recommendations**:
- As a general starting point, a temperature of .2, top-P of .95, and top-K of 30 will give you relatively coherent results that can be creative but not excessively so. 
- If you want especially creative results, try starting with a temperature of .9, top-P of .99, and top-K of 40. 
- If you want less creative results, try starting with a temperature of .1, top-P of .9, and top-K of 20.
- If your task always has a single correct answer (e.g., answering a math problem), start with a temperature of 0.

> NOTE: With more freedom (higher temperature, top-K, top-P, and output tokens), the LLM might generate text that is less relevant.
---