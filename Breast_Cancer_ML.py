import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ------------------------------------------------------------
# Display Border
# ------------------------------------------------------------

def Border():
    print("-" * 70)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

def LoadData():
    CancerData = load_breast_cancer()

    Data = pd.DataFrame(
        CancerData.data,
        columns=CancerData.feature_names
    )

    Data["target"] = CancerData.target

    return Data

# ------------------------------------------------------------
# Explore Dataset
# ------------------------------------------------------------

def ExploreData(Data):
    Border()
    print("Dataset Information")
    Border()
    print("Dataset Shape :", Data.shape)
    print("Number of Features :", Data.shape[1] - 1)
    print("Target Values :", Data["target"].unique())
    print("Missing Values :", Data.isnull().sum().sum())
    print("Duplicate Rows :", Data.duplicated().sum())

    Border()
    print("Summary Statistics")
    Border()
    print(Data.describe())

# ------------------------------------------------------------
# Prepare Data
# ------------------------------------------------------------

def PrepareData(Data):
    Features = Data.drop("target", axis=1)
    Target = Data["target"]

    if Features.isnull().sum().sum() > 0:
        Features = Features.fillna(Features.median())

    return Features, Target

# ------------------------------------------------------------
# Display Correlation
# ------------------------------------------------------------

def DisplayCorrelation(Data):
    Border()
    print("Feature Correlation Visualization")
    Border()

    Correlation = Data.drop("target", axis=1).corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(Correlation, cmap="coolwarm")
    plt.title("Breast Cancer Feature Correlation")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Split Dataset
# ------------------------------------------------------------

def SplitData(Features, Target):
    X_train, X_test, Y_train, Y_test = train_test_split(
        Features,
        Target,
        test_size=0.2,
        random_state=42,
        stratify=Target
    )

    return X_train, X_test, Y_train, Y_test

# ------------------------------------------------------------
# Scale Features
# ------------------------------------------------------------

def ScaleData(X_train, X_test):
    Scaler = StandardScaler()

    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------

def TrainModel(X_train, Y_train):
    Model = LogisticRegression(max_iter=10000)
    Model.fit(X_train, Y_train)
    return Model

# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

def EvaluateModel(Model, X_test, Y_test):
    Y_pred = Model.predict(X_test)

    Accuracy = accuracy_score(Y_test, Y_pred)
    Matrix = confusion_matrix(Y_test, Y_pred)
    Report = classification_report(Y_test, Y_pred)

    Border()
    print("Model Evaluation")
    Border()
    print("Accuracy :", Accuracy)
    print("Accuracy Percentage :", Accuracy * 100, "%")

    Border()
    print("Confusion Matrix")
    Border()
    print(Matrix)

    Border()
    print("Classification Report")
    Border()
    print(Report)

    return Y_pred

# ------------------------------------------------------------
# Display Observations
# ------------------------------------------------------------

def DisplayObservations(Data, Accuracy):
    Border()
    print("Observations and Conclusion")
    Border()
    print("The dataset contains", Data.shape[0], "records and", Data.shape[1] - 1, "features.")
    print("The target contains two classes: 0 represents Malignant and 1 represents Benign.")
    print("Feature scaling was performed using StandardScaler.")
    print("The classification model was trained and evaluated using separate training and testing data.")
    print("Testing Accuracy :", Accuracy * 100, "%")
    print("The confusion matrix and classification report were used to evaluate classification performance.")

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    Border()
    print("Breast Cancer Prediction")
    Border()

    Data = LoadData()

    print("Dataset Loaded Successfully")

    ExploreData(Data)

    Features, Target = PrepareData(Data)

    DisplayCorrelation(Data)

    X_train, X_test, Y_train, Y_test = SplitData(
        Features,
        Target
    )

    Border()
    print("Training and Testing Data")
    Border()
    print("Training Samples :", len(X_train))
    print("Testing Samples :", len(X_test))

    X_train_scaled, X_test_scaled = ScaleData(
        X_train,
        X_test
    )

    Border()
    print("Feature Scaling Completed Successfully")
    Border()

    Model = TrainModel(
        X_train_scaled,
        Y_train
    )

    Border()
    print("Classification Model Trained Successfully")
    Border()

    Y_pred = EvaluateModel(
        Model,
        X_test_scaled,
        Y_test
    )

    Accuracy = accuracy_score(
        Y_test,
        Y_pred
    )

    DisplayObservations(
        Data,
        Accuracy
    )

    Border()
    print("Breast Cancer Prediction Completed Successfully")
    Border()

if __name__ == "__main__":
    main()