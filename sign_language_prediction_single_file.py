'''
pip install onnxruntime
pip install numpy
pip install pandas
'''

#Load ONNX save model
import onnxruntime as rt

#Change data type
import numpy as np

#Read csv file
import pandas as pd

def predict_single_file (ANN_model, file_path):
    # Input and output values
    input_name = ANN_model.get_inputs()[0].name
    output_name = ANN_model.get_outputs()[0].name

    # Load test data (test.csv needs to be a single-row csv file)
    data = pd.read_csv(file_path, header=None) # header = None if there is not header for colum
    X_test = data.iloc[:, :-1].values
    X_test = X_test.astype(np.float32)

    # Number of class to letter label mapping
    labels = {
        "0": "a", "1": "b", "2": "c", "3": "d", "4": "e",
        "5": "f", "6": "g", "7": "h", "8": "i", "9": "k",
        "10": "l", "11": "m", "12": "n", "13": "o", "14": "p",
        "15": "q", "16": "r", "17": "s", "18": "t", "19": "u",
        "20": "v", "21": "w", "22": "x", "23": "y",
    }


    # Do the prediction
    pred = ANN_model.run([output_name], {input_name: X_test.astype(np.float32)})[0]
    pred = labels.get(str(pred[0]))

    # Imprimir las predicciones
    print(f"That letter is: {pred}")