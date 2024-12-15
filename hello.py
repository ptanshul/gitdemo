import pandas as pd
import sagemaker
from sagemaker import S3Input
from sagemaker.sklearn.estimator import SKLearn

# Download the Abalone dataset
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data

# Read the data into a Pandas DataFrame
df = pd.read_csv(
    "abalone.data", 
    names=[
        "Sex", "Length", "Diameter", "Height", "Whole weight", 
        "Shucked weight", "Viscera weight", "Shell weight", "Rings"
    ]
)

# Convert categorical feature 'Sex' to numerical using one-hot encoding
df = pd.get_dummies(df, columns=["Sex"], prefix=["Sex"])

# Split data into features (X) and target (y)
X = df.drop("Rings", axis=1)
y = df["Rings"]

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to CSV and upload to S3
train_data = pd.concat([y_train, X_train], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

bucket = sagemaker.Session().default_bucket()
prefix = "abalone-example"

train_data.to_csv("train.csv", header=False, index=False)
test_data.to_csv("test.csv", header=False, index=False)

boto3.Session().resource('s3').Bucket(bucket).Object(f'{prefix}/train/train.csv').upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(f'{prefix}/test/test.csv').upload_file('test.csv')

train_input = S3Input(s3_data=f's3://{bucket}/{prefix}/train', content_type='text/csv')
test_input = S3Input(s3_data=f's3://{bucket}/{prefix}/test', content_type='text/csv')