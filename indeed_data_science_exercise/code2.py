from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import csv as csv
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

dataset_path = 'data'
train_datafile_features = os.path.join(dataset_path, 'train_features_2013-03-07.csv')
train_datafile_salaries = os.path.join(dataset_path, 'train_salaries_2013-03-07.csv')
test_datafile_features = os.path.join(dataset_path, 'train_features_2013-03-07.csv')
train_data_features = pd.read_csv(train_datafile_features)
train_data_salaries = pd.read_csv(train_datafile_salaries)
test_data_features = pd.read_csv(test_datafile_features)
train_data = pd.merge(train_data_features, train_data_salaries, on=['jobId'], how='left')

california_housing_dataframe = train_data.reindex(
    np.random.permutation(train_data.index))


#USE THE ONEHOT ENCORDER TO TRANSFER LABELS
encoder = LabelBinarizer()
companyId_label = encoder.fit_transform(california_housing_dataframe["companyId"])
california_housing_dataframe["companyId"] = companyId_label
jobType_label = encoder.fit_transform(california_housing_dataframe["jobType"])
california_housing_dataframe["jobType"] = jobType_label
degree_label = encoder.fit_transform(california_housing_dataframe["degree"])
california_housing_dataframe["degree"] = degree_label
major_label = encoder.fit_transform(california_housing_dataframe["major"])
california_housing_dataframe["major"] = major_label
industry_label = encoder.fit_transform(california_housing_dataframe["industry"])
california_housing_dataframe["industry"] = industry_label
print (companyId_label.shape)
#jobType_label = encoder.fit_transform(training_examples["jobType"])

def preprocess_features(california_housing_dataframe):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = california_housing_dataframe[
    ["companyId",
     "jobType",
     "degree",
     "major",
     "industry",
     "yearsExperience",
     "milesFromMetropolis"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["salary"] = (
    california_housing_dataframe["salary"])
  return output_targets

  # Choose the first 750000 (out of 1000000) examples for training.
training_examples = preprocess_features(california_housing_dataframe.head(500000))
training_targets = preprocess_targets(california_housing_dataframe.head(500000))

# Choose the last 250000 (out of 1000000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(250000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(250000))

test_examples = preprocess_targets(california_housing_dataframe[500001:750000])
test_targets = preprocess_targets(california_housing_dataframe[500001:750000])

print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}   
                                           
    #input_fn = tf.estimator.inputs.numpy_input_fn(
    #                                              x=features, y=targets,
    #                                              batch_size=batch_size, shuffle=shuffle)
    #return input_fn
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]


def train_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    feature_columns: A `set` specifying the input feature columns to use.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  #my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
  
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["salary"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["salary"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["salary"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    print("  period %02d : %0.2f" % (period, validation_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()

  return linear_regressor


def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """
  
  companyId = tf.feature_column.numeric_column("companyId")
  jobType = tf.feature_column.numeric_column("jobType")
  degree = tf.feature_column.numeric_column("degree")
  major = tf.feature_column.numeric_column("major")
  industry = tf.feature_column.numeric_column("industry")
  yearsExperience = tf.feature_column.numeric_column("yearsExperience")
  milesFromMetropolis = tf.feature_column.numeric_column("milesFromMetropolis")


  bucketized_yearsExperience = tf.feature_column.bucketized_column(
    yearsExperience, boundaries=get_quantile_based_boundaries(
      training_examples["yearsExperience"], 10))
  bucketized_milesFromMetropolis = tf.feature_column.bucketized_column(
    milesFromMetropolis, boundaries=get_quantile_based_boundaries(
      training_examples["milesFromMetropolis"], 10))
  
  feature_columns = set([companyId,jobType,degree,major,industry,bucketized_yearsExperience,
                         bucketized_milesFromMetropolis])
  
  return feature_columns
feature_columns=construct_feature_columns()


linear_regressor = train_model(
    learning_rate=2.0,
    steps=500,
    batch_size=1000,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

#Evaluation

predict_test_input_fn = lambda: my_input_fn(
      test_examples, 
      test_targets["salary"], 
      num_epochs=1, 
      shuffle=False)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)


