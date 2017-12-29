import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def read(filename):
	df = pd.read_excel(filename, encoding = "ascii")
	Xvalues = df['X']
	Yvalues = df['Y']
	return np.array([np.array([x, y]) for x, y in zip(Xvalues, Yvalues)])

def build_linear_model(train_type):
	# Place Holders of TensorFlow are used for keeping the training examples
	X = tf.placeholder(tf.float32, name = "X")
	Y = tf.placeholder(tf.float32, name = "Y")

	# Variables of TensorFlow are used for keeping the trainable variables
	m = tf.Variable(0.0, name = "slope")
	c = tf.Variable(0.0, name = "Y-intercept")

	# Predict the Y's
	y_pred = m * X + c

	# Compute the mean squared loss
	if train_type == "B":
		loss = tf.reduce_mean(tf.square(Y - y_pred, name = "loss"))
	else:
		loss = tf.square(Y - y_pred, name = "loss")

	# Optimize the linear model
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

	return {"optimizer" : optimizer, "m" : m, "c" : c, "X" : X, "Y" : Y, "loss" : loss}

def plot(X, Y, m, c, losses, num_of_epochs):
	plt.subplot(2, 2, 1)
	plt.scatter(X, Y, color = "blue", s = 5)
	plt.plot(X, m*X + c, color = "red", linewidth = 0.5)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Regression Plot")

	plt.subplot(2, 2, 2)
	x_loss = np.array([x+1 for x in range(num_of_epochs)])
	plt.plot(x_loss, losses, color = "red", linewidth = 0.5)
	plt.xlabel("epoch number")
	plt.ylabel("loss")
	plt.title("Variation of loss function with respect to the number of epochs")

	plt.subplot(2, 2, 3)
	plt.scatter(X, Y - (m * X + c), color = "blue", s = 5)
	plt.plot(X, np.zeros_like(X), color = "green", linewidth = 0.5)
	plt.xlabel("fitted value")
	plt.ylabel("residuals")
	plt.title("Residual Plot")

	plt.subplot(2, 2, 4)
	plt.scatter(Y, m * X + c, color = "blue", s = 5)
	plt.xlabel("Y")
	plt.ylabel("Predicted Y")
	plt.title("Observed vs Predicted")

	plt.show()

def train_model(regressor, num_of_epochs, train_type, data):
	losses = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_of_epochs):
			if train_type == "B": # Batch Training
				_, loss = sess.run([regressor["optimizer"], regressor["loss"]], feed_dict = {regressor["X"]:data[:, 0], regressor["Y"]:data[:, 1]})
				losses.append(loss)
			else:
				loss = 0
				for x, y in data: # Online Training
					_, temp = sess.run([regressor["optimizer"], regressor["loss"]], feed_dict = {regressor["X"]:x, regressor["Y"]:y})
					loss += temp
				loss /= len(data)
				losses.append(loss)
		slope, y_intercept = sess.run([regressor["m"], regressor["c"]])

		if train_type == "B":
			writer = tf.summary.FileWriter('./batch_training_linear', sess.graph)
		else:
			writer = tf.summary.FileWriter('./online_training_linear', sess.graph)

	return slope, y_intercept, losses

def main(train_type):
	dataset_path = "Data/dataset.xls"
	num_of_epochs = 100

	data = read(dataset_path)
	linear_regres = build_linear_model(train_type)
	slope, y_intercept, losses = train_model(linear_regres, num_of_epochs, train_type, data)

	print(slope, y_intercept)
	plot(data[:, 0], data[:, 1], slope, y_intercept, np.array(losses), num_of_epochs)

if __name__ == '__main__':
	print("Batch Training or Online Training? Enter \"B\" for Batch and \"O\" for Online.")
	train_type = input()
	main(train_type)