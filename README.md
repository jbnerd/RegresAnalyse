# Regression Analysis

- This repository contains the code written in Python's TensorFlow library for Linear Regression Analysis over a miniscule dummy dataset. It was written during the process of learning the TensorFlow library from scratch.

## Dependencies

- Python3
- Numpy
- TensorFlow
- MatPlotLib
- Pandas

## Dataset

- Find the dataset [here](http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr05.html).
- You can also find the dataset in the `Dataset` directory within this repository.

### Description of the Dataset

- Name: Fire and Theft in Chicago
- X = fires per 1000 housing units
- Y = thefts per 1000 population within the same Zip code in the Chicago metro area
- Total number of Zip code areas: 42

## Running the code

- Type in `python3 Regres.py` in the terminal to run the code.

## Visulizing the Computational Graph

- Once you have run the code on your machine, type in `tensorboard --logdir="<path_to_summaries>"` in your terminal.
	- `<path_to_summaries>` = `./batch_training_linear` if you chose the Batch Training mode.
	- `<path_to_summaries>` = `./online_training_linear` if you chose the Online Training mode.

## Example Plots

![alt text](https://github.com/jbnerd/RegresAnalyse/blob/master/Batch_linear_model.png)