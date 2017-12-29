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
	plt.subplot(2, 1, 1)
	plt.scatter(X, Y, color = "blue", s = 5)
	plt.plot(X, m*X + c, color = "red", linewidth = 0.5)
	plt.xlabel("X")
	plt.ylabel("Y")

	plt.subplot(2, 1, 2)
	x_loss = np.array([x+1 for x in range(num_of_epochs)])
	plt.plot(x_loss, losses, color = "red", linewidth = 0.5)
	plt.xlabel("epoch number")
	plt.ylabel("loss")

	plt.show()

def main(train_type):
	dataset_path = "Data/dataset.xls"
	num_of_epochs = 100

	data = read(dataset_path)
	linear_regres = build_linear_model(train_type)
	losses = []

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_of_epochs):
			if train_type == "B": # Batch Training
				_, loss = sess.run([linear_regres["optimizer"], linear_regres["loss"]], feed_dict = {linear_regres["X"]:data[:, 0], linear_regres["Y"]:data[:, 1]})
				losses.append(loss)
			else:
				loss = 0
				for x, y in data: # Online Training
					_, temp = sess.run([linear_regres["optimizer"], linear_regres["loss"]], feed_dict = {linear_regres["X"]:x, linear_regres["Y"]:y})
					loss += temp
				loss /= len(data)
				losses.append(loss)
		slope, y_intercept = sess.run([linear_regres["m"], linear_regres["c"]])

	print(slope, y_intercept)
	plot(data[:, 0], data[:, 1], slope, y_intercept, np.array(losses), num_of_epochs)

if __name__ == '__main__':
	print("Batch Training or Online Training? Enter \"B\" for Batch and \"O\" for Online.")
	train_type = input()
	main(train_type)