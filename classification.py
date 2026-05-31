from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from colorama import Fore, init

init()

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# KNN Model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
knn_predictions = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

# Menu Loop
while True:

    print("\n" + "=" * 50)
    print("IRIS FLOWER CLASSIFICATION USING AI")
    print("=" * 50)

    print("1. Show Accuracy")
    print("2. Show Classification Report")
    print("3. Predict Flower")
    print("4. Show Accuracy Graph")
    print("5. Exit")

    choice = input("\nEnter your choice: ")

    # Option 1
    if choice == "1":

        print(Fore.GREEN +
              f"\nDecision Tree Accuracy: {round(accuracy*100,2)} %")

        print(Fore.BLUE +
              f"KNN Accuracy: {round(knn_accuracy*100,2)} %")

        if accuracy > knn_accuracy:
            print("\nBest Model: Decision Tree")

        elif knn_accuracy > accuracy:
            print("\nBest Model: KNN")

        else:
            print("\nBest Model: Both models performed equally well")

    # Option 2
    elif choice == "2":

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        print("\nConfusion Matrix:")
        print(cm)

    # Option 3
    elif choice == "3":

        print("\nEnter Flower Measurements")

        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))

        user_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        prediction = model.predict(user_data)

        flower_name = iris.target_names[prediction[0]]

        print("\nPredicted Flower:", flower_name)

        print("\nFlower Categories:")
        print(iris.target_names)

    # Option 4
    elif choice == "4":

        models = ["Decision Tree", "KNN"]
        accuracies = [accuracy * 100, knn_accuracy * 100]

        plt.bar(models, accuracies)
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy (%)")
        plt.show()

    # Option 5
    elif choice == "5":

        print("\nThank You!")
        break

    # Invalid Choice
    else:

        print("\nInvalid Choice! Please enter 1 to 5.")