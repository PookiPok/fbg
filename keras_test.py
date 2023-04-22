import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x = tf.ones(shape=(2,1))
print(x)


num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(mean=[0, 3],cov=[[1, 0.5],[0.5, 1]],size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(mean=[3, 0],cov=[[1, 0.5],[0.5, 1]],size=num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
np.ones((num_samples_per_class, 1), dtype="float32")))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))