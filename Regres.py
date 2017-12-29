import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def read(filename):
	df = pd.read_excel(filename, encoding = "ascii")
	Xvalues = df['X']
	Yvalues = df['Y']
	return np.array([np.array([x, y]) for x, y in zip(Xvalues, Yvalues)])

def build_linear_model():
	# Place Holders of TensorFlow are used for keeping the training examples
	X = tf.placeholder(tf.float32, name = "X")
	Y = tf.placeholder(tf.float32, name = "Y")

	# Variables of TensorFlow are used for keeping the trainable variables
	m = tf.Variable(0.0, name = "slope")
	c = tf.Variable(0.0, name = "Y-intercept")

	# Predict the Y's
	y_pred = m * X + c

	# Compute the mean squared loss
	loss = tf.square(Y - y_pred, name = "loss")

	# Optimize the linear model
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

	return {"optimizer" : optimizer, "m" : m, "c" : c, "X" : X, "Y" : Y, "loss" : loss}

def main():
	dataset_path = "Data/dataset.xls"
	data = read(dataset_path)
	linear_regres = build_linear_model()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(100):
			print(i)
			for x, y in data:
				sess.run(linear_regres['optimizer'], feed_dict = {linear_regres["X"]:x, linear_regres["Y"]:y})
		slope, y_intercept = sess.run([linear_regres["m"], linear_regres["c"]])

	print(slope, y_intercept, linear_regres["loss"])

if __name__ == '__main__':
	main()