import multilayer_perceptron_training as mpt
import sign_language_images_to_csv as slitc
import sign_language_prediction_single_file as slpsf
import onnxruntime as rt
import sign_language_prediction_real_time as slprt


sess = None

def make_new_csv():
    print("Select your database directory: ")
    database_path = input()
    slitc.process_images_from_folders(database_path)
    print("CSV file created.")

def create_and_train_ANN():
    print("Select your CSV file: ")
    csv_path = input()
    print("How many data do you want to use for testing?: ")
    test_size = input()
    mpt.create_and_train(csv_path, float(test_size))
    print("ANN model created.")

def load_ANN():
    global sess
    print("Select your ONNX file: ")
    onnx_file = input()
    sess = rt.InferenceSession(onnx_file)
    print("ANN model loaded successfully")

def test_ANN():
    print("Select your test path: ")
    test_path = input()
    print("Select your test file name: ")
    file_name = input()
    slitc.process_image_from_route(test_path, file_name)
    slpsf.predict_single_file(sess, './predict_landmarks.csv')

def test_ANN_real_time():
    import sign_language_prediction_real_time as slprt
    slprt.get_ANN(sess)

while True:
    print("\n")
    print("Sign Language Recognizer")
    print("0. Exit")
    print("1. Make a new CSV.")
    print("2. Create and train a new ANN.")
    print("3. Load a trained ANN model.")
    print("4. Test loaded ANN model.")
    print("5. Test loaded ANN model in real time.")
    print("Choose your option:")
    userAction = input()

    if userAction == '5':
        if sess is not None:
            test_ANN_real_time()
        else:
            print("There is no loaded ANN model yet.")
    elif userAction == '4':
        if sess is not None:
            test_ANN()
        else:
            print("There is no loaded ANN model yet.")
    elif userAction == '3':
        load_ANN()
    elif userAction == '2':
        create_and_train_ANN()
    elif userAction == '1':
        make_new_csv()
    elif userAction == '0':
        break
    else:
        print("No valid option.")