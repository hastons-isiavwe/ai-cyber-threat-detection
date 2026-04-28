from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Drop unnamed columns and columns with all NaN values
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data.dropna(axis=1, how='all')  # Drop columns with all NaN values

    # Drop irrelevant columns like 'Circuit', if present
    data = data.drop(columns=['Circuit'], errors='ignore')

    # Encode categorical labels (e.g., 'Trojan Free' and 'Trojan Infected')
    if 'Label' in data.columns:
        le = LabelEncoder()
        data['Label'] = le.fit_transform(data['Label'])
    
    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    data.iloc[:, :] = imputer.fit_transform(data)
    
    # Split features and target labels
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balance the training dataset using SMOTE
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test
