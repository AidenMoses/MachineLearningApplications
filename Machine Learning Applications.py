# Import necessary libraries
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set seaborn style for plots
sns.set()

# Question 3: K-Means Clustering
# Load data from a CSV file
Data = pd.read_csv('Data.csv')

# Create a basic scatter plot chart as a base
plt.scatter(Data['X'], Data['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Copy the CSV data
x = Data.copy()

# Define a function for plotting clusters
def Plot(num):
    kmeans = KMeans(num)
    kmeans.fit(x)
    cluster = x.copy()
    cluster['cluster_pred'] = kmeans.fit_predict(x)
    
    # Plot the scatter chart with clusters
    plt.scatter(cluster['X'], cluster['Y'], c=cluster['cluster_pred'], cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # Print cluster centers
    point = kmeans.cluster_centers_
    point_r = np.round(point, 2)
    print(point_r)

# Plot clusters for different numbers of clusters
Plot(3)
Plot(4)
Plot(5)
Plot(6)

# Q3: Observations
# When the number of clusters increases, some clusters split into multiple subclusters, leading to new centroids.

# Question 4: The Elbow Method
# Scale the data
x_scaled = preprocessing.scale(x)

# Calculate within-cluster sum of squares (WCSS) for different cluster numbers
wcss = []
for i in range(1, 30):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

# Plot the WCSS for different cluster numbers
plt.plot(range(1, 30), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Q4: Observations
# The elbow method suggests that the optimal number of clusters is three, which is different from the initial guess of five.

# Print clusters with the recommended number of clusters
kmeans_new = KMeans(3)
kmeans.fit(x_scaled)
cluster_new = x.copy()
cluster_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
cluster_new

# Plot clusters with the recommended number of clusters
plt.scatter(cluster_new['X'], cluster_new['Y'], c=cluster_new['cluster_pred'], cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Question 5: Adding Data
# Create a new DataFrame with additional data
a = pd.DataFrame({"X": [55, 26, 40, 40], "Y": [38, 82, 38, 10]})
Data2 = Data.append(a, ignore_index=True)
Data2

# Copy the new data
x2 = Data2.copy()

# Scale the new data
x_scaled2 = preprocessing.scale(x2)

# Train KMeans with the recommended number of clusters
kmeans_new = KMeans(3)
kmeans.fit(x_scaled2)
cluster2 = x2.copy()
cluster2['cluster_pred'] = kmeans_new.fit_predict(x_scaled2)

# Plot clusters with the appended data
plt.scatter(cluster2['X'], cluster2['Y'], c=cluster2['cluster_pred'], cmap='rainbow')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Q5: Observations
# KMeans effectively clusters the data, including the appended data points.

# Question 6: Cluster with Appended Data
# Create a simple Perceptron Neural Network
# Input data
inputs = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1]])

# Output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    def backpropagation(self):
        self.error = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    def train(self, epochs=25000):
        for epoch in range(epochs):
            self.feed_forward()
            self.backpropagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# Create and train the neural network
NN = NeuralNetwork(inputs, outputs)
NN.train()

# Example input data for prediction
example = np.array([[1, 1, 0]])
example_2 = np.array([[0, 1, 1]])

# Print predictions for the examples
print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example_2), ' - Correct: ', example_2[0][0])

# Plot the error over training epochs
plt.figure(figsize=(15, 5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

# Q6: Observations
# The neural network is trained successfully and makes predictions for the given examples.

# Sample 1
# New input data
inputs1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

# Output data
outputs1 = np.array([[0], [0], [0], [1], [1], [1]])

# Create and train a new neural network with sample 1
NN1 = NeuralNetwork(inputs1, outputs1)
NN1.train()

# Print predictions for the examples
print(NN1.predict(example), ' - Correct: ', example[0][0])
print(NN1.predict(example_2), ' - Correct: ', example_2[0][0])

# Plot the error over training epochs
plt.figure(figsize=(15, 5))
plt.plot(NN1.epoch_list, NN1.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

# Sample 2
# New input data
inputs2 = np.array([[1
