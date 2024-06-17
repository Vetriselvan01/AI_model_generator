import pandas as pd
import numpy as np
import csv
import joblib
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error,accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import RidgeClassifier
def create():
  location = input("Enter dataset location : ")
  df = pd.read_csv(location)
  null_values = df.isnull().sum()
  df.fillna(0, inplace=True)
  non_integer_cols = df.select_dtypes(exclude='number')
  # Initialize the LabelEncoder
  le = LabelEncoder()
  # Convert string-type columns to integer type
  for col in non_integer_cols.columns.tolist():
      df[col] = le.fit_transform(df[col])
      integer_mapping = {label: encoded_value for encoded_value, label in enumerate(le.classes_)}
      print(integer_mapping)
  # split the data(independent & dependent)
  num_columns = count_columns(location)
  x = df.iloc[:,0:num_columns-1].values
  y= df.iloc[:,num_columns-1:num_columns].values
  xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 0)
  bin = check_binary_or_continuous(ytest)
  name = input("Enter the model name : ")
  alg_scores = {
      "linearreg": linearreg(xtrain, ytrain, xtest, ytest,0,bin,name),
      "randfor": randfor(xtrain, ytrain, xtest, ytest,0,bin,name),
      "graboost": graboost(xtrain, ytrain, xtest, ytest,0,bin,name)
  }

  alg = max(alg_scores, key=alg_scores.get)
  r2score = alg_scores[alg]

  alg_functions = {
      "linearreg": linearreg,
      "randfor": randfor,
      "graboost": graboost
  }

  # Call the appropriate function based on the algorithm name
  r2 = alg_functions[alg](xtrain, ytrain, xtest, ytest,1,bin,name)
  filename = "models.txt"
  data = name+".pkl"
  write_to_text_file(filename, data)
def count_columns(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # Assuming the first row has the headers
        first_row = next(reader)
        num_columns = len(first_row)
    return num_columns

def flatten_list(lst):
    """Flatten a nested list."""
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened

def check_binary_or_continuous(ytest):
    # Convert numpy array to a list
    ytest_list = ytest.tolist() if isinstance(ytest, np.ndarray) else ytest

    # Flatten the list
    ytest_flattened = flatten_list(ytest_list)

    # Convert flattened list to tuple (hashable)
    ytest_tuple = tuple(ytest_flattened)

    unique_values = set(ytest_tuple)

    # Check if the unique values are only 0 and 1
    if set(unique_values) == {0, 1}:
        return 1
    else:

        return 0

# Assuming xtest is a numpy array or a list containing your test data
# Replace 'xtest' with the actual name of your test data variable

def linearreg(xtrain,ytrain,xtest,ytest,save,bin,name):
  #linear  regression

  # Initialize the Linear Regression model
  model = LinearRegression()

  # Train the model
  model.fit(xtrain, ytrain)

  # Make predictions on the test set
  predictions = model.predict(xtest)

  # Calculate the R2 score
  r2 = r2_score(ytest, predictions)
  if(save==1):
    joblib.dump(model, name+'.pkl')
    evaluation(ytest,predictions,bin)
  return r2
def graboost(xtrain,ytrain,xtest,ytest,save,bin,name):
  if(bin):
    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,  # Reduced from 100 to prevent overfitting
        min_samples_split=10,  # Adjusted for regularization
        min_samples_leaf=5,  # Adjusted for regularization
        subsample=0.8,  # Adjusted for regularization
        random_state=0
    )
  else:
      model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,  # Reduced from 100 to prevent overfitting
        min_samples_split=10,  # Adjusted for regularization
        min_samples_leaf=5,  # Adjusted for regularization
        subsample=0.8,  # Adjusted for regularization
        random_state=0
      )

  # Train the model
  model.fit(xtrain, ytrain.ravel())

  # Make predictions on the test set
  predictions = model.predict(xtest)

  if(bin):
    accuracy = accuracy_score(ytest, predictions)
  else:
    r2 = r2_score(ytest, predictions)
  if(save==1):
    joblib.dump(model, name+'.pkl')
    evaluation(ytest,predictions,bin)
  if(bin):
    return accuracy
  else:
    return r2
def randfor(xtrain,ytrain,xtest,ytest,save,bin,name):
  #random forest
  # Initialize Random Forest Regression model
  if(bin):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
  else:
    model = RandomForestRegressor(n_estimators=100, random_state=0)

  # Train the model
  model.fit(xtrain, ytrain.ravel())

  # Make predictions on the test set
  predictions = model.predict(xtest)

  # Calculate the R2 score
  if(bin):
    accuracy = accuracy_score(ytest, predictions)
  else:
    r2 = r2_score(ytest, predictions)
  if(save==1):
    joblib.dump(model, name+'.pkl')
    evaluation(ytest,predictions,bin)
  if(bin):
    return accuracy
  else:
    return r2

def evaluation(ytest,predictions,bin):
  # R-squared (R2 score)
  if(bin):
    accuracy = accuracy_score(ytest, predictions)

    # Print confusion matrix and classification report
    print("Accuracy: ",accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix(ytest, predictions))
    print("\nClassification Report:")
    print(classification_report(ytest, predictions))

  else:
    r2 = r2_score(ytest,predictions)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(ytest,predictions)

    # Mean Squared Logarithmic Error (MSLE)
    msle = mean_squared_log_error(ytest,predictions)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(ytest,predictions)

    print("R-squared (R2 score):", r2)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Squared Logarithmic Error (MSLE):", msle)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Percentage Error (MAPE):", mape)
def write_to_text_file(filename, data):
    with open(filename, 'a') as file:
        file.write(data + '\n')
def read_text_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def print_numbered_list(data):
    for i, item in enumerate(data, 1):
        print(f"{i}.{item}")
def load(load_model):
  # Load the saved model
  base,extension = load_model.rsplit(".",1)
  filename = base+".csv"
  with open(filename, 'r', newline='') as csvfile:
          reader = csv.reader(csvfile)
          columns = next(reader)  # Get the column names from the first row

  columns = columns[0:-1]
  new_row = get_row_input(columns)
  new_row = np.array(new_row).reshape(1, -1)
  loaded_model = joblib.load(load_model)
  prediction = loaded_model.predict(new_row)
  print(base + " value : ",prediction)
  with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(new_row)
def pretrain(load_model):
  # Load the saved model
  base,extension = load_model.rsplit(".",1)
  filename = base+".csv"
  with open(filename, 'r', newline='') as csvfile:
          reader = csv.reader(csvfile)
          columns = next(reader)  # Get the column names from the first row

  columns = columns[0:-1]
  new_row = get_row_input(columns)
  new_row = np.array(new_row).reshape(1, -1)
  loaded_model = joblib.load(load_model)
  prediction = loaded_model.predict(new_row)
  if(base == "weather prediction"):
    weather_mapping = {0:'drizzle', 1:'fog', 2:'rain', 3:'snow', 4:'sun'}
    prediction = prediction.item()
    selected = weather_mapping.get(prediction, None)
    if selected is not None:
      prediction = selected
  if(base == "cancer prediction"):
    weather_mapping = {0:'high probability', 1:'low probability', 2:'medium probability'}
    prediction = prediction.item()
    selected = weather_mapping.get(prediction, None)
    if selected is not None:
      prediction = selected
  print(base +" : ",prediction)
def get_row_input(columns):
    row_data = []
    for column in columns:
        value = input(f"Enter value for '{column}': ").strip()
        row_data.append(float(value))
    return row_data
def main():
      while True:
        filename = "models.txt"
        data = read_text_file(filename)
        print("0.create model\n")
        print_numbered_list(data)


        choice = input("Enter the number of the option you want to select (or 'q' to quit): ")
        if choice.lower() == 'q':
            break
        if choice == '0':
          create()
          continue
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= 5:
                pretrain(data[choice_num - 1])
                continue
            elif 6 <= choice_num <= len(data):
              load(data[choice_num - 1])
              continue
            else:
                print("Invalid option number.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")


if __name__ == "__main__":
    main()
