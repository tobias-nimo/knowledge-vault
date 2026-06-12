## Data Snooping Bias
Before exploring any dataset in depth, you should create a **test set**, set it aside, and never use it during development.

The reason is that humans are excellent at finding patterns. If you repeatedly look at the test set, you may unconsciously make decisions based on information it contains, such as choosing a particular model, feature set, or hyperparameters because they seem to perform well on that data.

As a result, the test set is no longer truly "unseen." **Performance measured on it will be overly optimistic** and will not accurately reflect how the model will perform on new data. This phenomenon is known as data snooping bias.