import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model_location = os.environ["MODEL_LOCATION"]

print("Fetching model from %s" %(model_location))

# load the model from disk
loaded_model = pickle.load(open(model_location, 'rb'))

def convert_array(arr):
    """
    Helper function to convert array for model inference
    """
    if isinstance(arr, list) and len(arr)==7:
        D = {}
        for ind, i in enumerate(arr):
            D[f"x{ind+1}"] = [i]
        return pd.DataFrame.from_dict(D)
    else:
        return None


@app.route('/score',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    x = convert_array(data)
    if x is not None:
        prediction = loaded_model.predict(x).tolist()
        print('Predictions: ',prediction)
        return jsonify(prediction)
    else:
        return jsonify("Unable to make prediction. Please check data input format")


@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(port=8887, debug=True, host='0.0.0.0')


# # dummy dataframe for local testing
# x = pd.DataFrame.from_dict({'x1':[5.605362,5.605362],
#                              'x2':[21.510036, 21.510036], 
#                              'x3':['Tue', 'Wed'], 
#                              'x4':[-2.687724, -2.687724], 
#                              'x5':[100.359864, 100.359864], 
#                              'x6':['New York', 'New York'], 
#                              'x7':['ford', 'ford']})