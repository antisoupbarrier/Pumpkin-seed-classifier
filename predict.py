import pickle
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_LR=0.075.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('pumpkin')

@app.route('/predict', methods=['POST'])
def predict():
    seed = request.get_json()
    print(seed)

    s = np.load('std.npy')
    m = np.load('mean.npy')

    seed = np.array(list(seed.values())).reshape(1, -1)
    
    seed = ((seed-m))/s
    
    print(seed)
    y_pred = model.predict(seed)
    seed_type = y_pred >= 0.5

    result = {
        'seed_type_prediction': float(y_pred),
        'Ürgüp_Sivrisi': bool(seed_type)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)