#Import library
import pandas as pd

#Import modules and submodules from sklearn
from sklearn.model_selection import train_test_split #Used to split data into 2 sets: one for training and another for testing
from sklearn.neural_network import MLPClassifier #Classifier based on a MLP
from sklearn.preprocessing import LabelEncoder #Used to convert categorical labels (text) into numerical values
from sklearn.metrics import accuracy_score, classification_report #calculates the accuracy of the model, function that generates a 
                                                                # report with various performance metrics

#Specify the path to your CSV file
data = pd.read_csv('hand_landmarks_total.csv')

#Separate the labels and characteristics
X = data.iloc[:, :-1] # All the columns except de lastone
y = data.iloc[:, -1] # The last one column 

print("FILAS: ", y)

# Encode the labels that are text to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Split data in sets of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and train de neuronal network
mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

#Make prediccions in training set
y_pred = mlp.predict(X_test)

 # Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Precisión del modelo:", accuracy)
print("\nReporte de clasificación:\n", report)


