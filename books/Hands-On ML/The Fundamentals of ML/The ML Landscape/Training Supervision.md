# Training Supervision
Machine Learning systems can be classified according to the amount and type of supervision they get during training.

## Supervised learning
In #supervised-learning, the training set you feed to the algorithm includes the desired solutions, called **labels** or **targets**.

![[1-5.png]]

Typical supervised learning tasks are: 
- **Classification**—assigning labels to new instances based on a labeled training set (see Figure 1-5).
- **Regression**—predicting a target numeric value, such as the price of a car, given a set of features (mileage, age, brand, etc.).

## Unsupervised learning
In #unsupervised-learning, as you might guess, the **training data is unlabeled**. The system tries to learn without a teacher.

![[1-7.png]]

Common unsupervised learning tasks are: 
- **Clustering**—detecting groups of similar entities (see Figure 1-7).
- **Anomaly detection**—identifying unusual instances, such as fraudulent credit card transactions.
- **Novelty detection**—it aims to detect new instances that look different from all instances in the training set.
- **Visualization algorithms**—transforming complex, unlabeled data into a 2D or 3D representation that can be easily plotted and explored.
- **Dimensionality reduction**—the goal is to simplify the data without losing too much information. One way to do this is to merge several correlated features into one. This is called feature extraction.
- **Association rule learning**—the goal is to dig into large amounts of data and discover interesting relations between attributes.

## Semi-supervised learning
Since labeling data is usually time-consuming and costly, you will often have plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that’s **partially labeled**. This is called #semi-supervised-learning.

![[1-10.png]]

Some photo-hosting services, such as **Google Photos**, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just add one label per person and it is able to name everyone in every photo, which is useful for searching photos.

Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms. For example, a clustering algorithm may be used to group similar instances together, and then every unlabeled instance can be labeled with the most common label in its cluster. Once the whole dataset is labeled, it is possible to use any supervised learning algorithm.

## Self-supervised learning
Another approach to machine learning involves actually **generating a fully labeled dataset from a fully unlabeled one**. Again, once the whole dataset is labeled, any supervised learning algorithm can be used. This approach is called #self-supervised-learning.

![[1-11.png]]

For example, if you have a large dataset of unlabeled images, you can randomly mask a small part of each image and then train a model to recover the original image. During training, the masked images are used as the inputs to the model, and the original images are used as the labels.

The resulting model may be quite useful in itself—for example, to repair damaged images or to erase unwanted objects from pictures. But more often than not, a model trained using self-supervised learning is not the final goal. You’ll usually want to tweak and fine-tune the model for a slightly different task—one that you actually care about.

Let's imagine you want to build a pet-species classifier but only have many unlabeled pet photos, you can **first use self-supervised learning to train a model on a related task**, such as repairing damaged images. To successfully reconstruct missing parts of an image, the model must learn meaningful features about different animals. **After this pre-training stage, the model can be fine-tuned on a smaller labeled dataset** so it learns the mapping between the pet species it already recognizes and the desired output labels.

> [!note]
> Transferring knowledge from one task to another is called #transfer-learning, and it’s one of the most important techniques in machine learning today, especially when using deep neural networks.

## Reinforcement learning
Here the learning system, called an **agent**, can observe the **environment**, select and perform **actions**, and get **rewards** in return (or penalties in the form of negative rewards). It must then learn by itself what is the best strategy, called a **policy**, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

![[1-12.png]]

DeepMind’s **AlphaGo** program is a good example of #reinforcement-learning.