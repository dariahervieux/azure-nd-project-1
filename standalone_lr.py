from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

""" Cleans the data. Returns cleaned data. """
def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    # The y column indicates if a customer subscribed to a fixed term deposit
    x_df["y"] = x_df.y.apply(lambda s: 1 if s == "yes" else 0)

    return x_df

""" Splits the data to data used for training and labels to predict. Returns a tuple (data, labels)"""
def split_train_label_data(x_df):
    # The y column indicates if a customer subscribed to a fixed term deposit
    y_df = x_df.pop("y")

    return (x_df, y_df)

def main():
    csv_path ='bankmarketing_train.csv'
    df = pd.read_csv(csv_path)

    # check classes balance
    print (df['y'].value_counts()/df.shape[0])

    x = clean_data(df)

    x, y = split_train_label_data(x)


    # Split data into train and test sets: 20% of the dataset to include in the test split.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train a model with the parameters found by HyperDrive
    model = LogisticRegression(C=119.04915426, max_iter=200).fit(x_train, y_train)

    # Weighted model
    #model = LogisticRegression(C=119.04915426, max_iter=200,class_weight={0:11, 1:89}).fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = model.score(x_test, y_test)

    print(f'Accuracy Score: {accuracy}')
    print(f'Balanced accuracy Score: {balanced_accuracy_score(y_test,y_pred)}')

    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification report:\n{classification_report(y_test,y_pred)}')

    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')

if __name__ == '__main__':
    main()