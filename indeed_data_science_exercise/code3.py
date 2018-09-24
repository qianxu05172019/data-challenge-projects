# import libaries
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
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# import data and preprocessing the data
dataset_path = 'data'
train_datafile_features = os.path.join(dataset_path, 'train_features_2013-03-07.csv')
train_datafile_salaries = os.path.join(dataset_path, 'train_salaries_2013-03-07.csv')
test_datafile_features = os.path.join(dataset_path, 'train_features_2013-03-07.csv')
train_data_features = pd.read_csv(train_datafile_features)
train_data_salaries = pd.read_csv(train_datafile_salaries)
test_data_features = pd.read_csv(test_datafile_features)
train_data = pd.merge(train_data_features, train_data_salaries, on=['jobId'], how='left')

salary = list(train_data.salary)
index = salary.index(0)
while 0 in salary:
  salary.remove(0)
train_data=train_data[train_data.salary.isin(salary)]
trainData_dataframe = train_data.reindex(
    np.random.permutation(train_data.index))




#USE THE ONEHOT ENCORDER TO TRANSFER catergorical features
def Labelencorder(data):
  encoder = LabelEncoder()
  companyId_label = encoder.fit_transform(data["companyId"])
  data["companyId"] = companyId_label
  jobType_label = encoder.fit_transform(data["jobType"])
  data["jobType"] = jobType_label
  degree_label = encoder.fit_transform(data["degree"])
  data["degree"] = degree_label
  major_label = encoder.fit_transform(data["major"])
  data["major"] = major_label
  industry_label = encoder.fit_transform(data["industry"])
  data["industry"] = industry_label
  return data

trainData_dataframe = Labelencorder(trainData_dataframe)
test_data_features = Labelencorder(test_data_features)
test_examples = test_data_features



def preprocess_features(data_dataframe):
  """Prepares input features from California housing data set.

  Args:
    data_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  
  selected_features = data_dataframe[
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

def preprocess_targets(data_dataframe):
  """Prepares target features (i.e., labels) from data set.

  Args:
    data_dataframe: A Pandas DataFrame expected to contain data
      from the training data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["salary"] = (
    data_dataframe["salary"])
  return output_targets

  # Choose the first 750000 (out of 1000000) examples for training.
training_examples = preprocess_features(trainData_dataframe.head(750000))
training_targets = preprocess_targets(trainData_dataframe.head(750000))

# Choose the last 250000 (out of 1000000) examples for validation.
validation_examples = preprocess_features(trainData_dataframe.tail(250000))
validation_targets = preprocess_targets(trainData_dataframe.tail(250000))


print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
  processed_features = examples_dataframe

  processed_features["yearsExperience"] = linear_scale(examples_dataframe["yearsExperience"])
  processed_features["milesFromMetropolis"] = linear_scale(examples_dataframe["milesFromMetropolis"])
  
  return processed_features

normalized_dataframe = normalize_linear_scale(preprocess_features(trainData_dataframe))
normalized_training_examples = normalized_dataframe.head(750000)
normalized_validation_examples = normalized_dataframe.tail(250000)

#plt.hist(normalized_training_examples["milesFromMetropolis"])
#plt.show()



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
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
def my_input_fn1(features, batch_size=1, shuffle=True, num_epochs=None):
    #return features for next batch
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}   
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices(features) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features = ds.make_one_shot_iterator().get_next()
    return features
predict_test_input_fn = lambda: my_input_fn1(
      test_examples, 
      num_epochs=1, 
      shuffle=False)


def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` to use as input features for training.
    training_targets: A `DataFrame`  use as target for training.
    validation_examples: A `DataFrame` to use as input features for validation.
    validation_targets: A `DataFrame` to use as target for validation.
      
  Returns:
    A tuple `(estimator, training_losses, validation_losses)`:
      estimator: the trained `DNNRegressor` object.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer,
      dropout = 0.25
  )
  
  # Create input functions.
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
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
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

  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])



dnn_regressor = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.15),
    steps=3000,
    batch_size=1000,
    #hidden_units=[256,128,64],
    hidden_units=[256,128,64],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

#Evaluation

predict_test_input_fn = lambda: my_input_fn1(
      test_examples, 
      num_epochs=1, 
      shuffle=False)

test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)

test_predictions = np.array([item['predictions'][0] for item in test_predictions])

DataFrame = pd.DataFrame(test_predictions)
DataFrame.to_csv("test.csv",index=False,sep=',')
