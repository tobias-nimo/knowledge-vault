# Batch vs. Online Learning

## Batch learning
In #batch-learning, a system **cannot learn from new data** once it has been trained. As a result, the model is typically trained before deployment and then runs without further learning; this is known as **offline learning**. Because the model must be trained on all available data, this approach generally requires significant time and computing resources.

Unfortunately, a model's performance tends to degrade over time because the world continues to evolve while the model remains unchanged. This phenomenon is often called #data-drift (or **model rot**). To maintain performance, the model must be retrained periodically using up-to-date data.

To incorporate new data, a new version of the model must be trained from scratch using the entire available dataset—not only the new data but also the old data—and then deployed to replace the existing model.

> Random Forests are an example of a batch learning algorithm.

## Online learning
In #online-learning, a system **can learn incrementally** by feeding it new data instances sequentially, either individually or in small groups called **mini-batches**. Each learning step is fast and inexpensive, allowing the system to learn from new data as it arrives.

A major challenge of online learning is that the system can quickly learn from **bad or malicious data**, causing its performance to deteriorate. To mitigate this risk, the system should be **closely monitored**, learning should be disabled if performance drops, and abnormal input data should be detected using techniques such as anomaly detection.

> **Gradient Descent** is by far the most common online learning algorithm.
