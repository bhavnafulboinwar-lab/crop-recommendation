from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
mx = pickle.load(open("model/minmaxscaler.pkl", "rb"))
sc = pickle.load(open("model/standscaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        data = np.array([[N,P,K,temp,humidity,ph,rainfall]])

        data = mx.transform(data)
        data = sc.transform(data)

        prediction = model.predict(data)
        print("Prediction =", prediction)

        crop_dict = {
            1:"Rice",
            2:"Maize",
            3:"Jute",
            4:"Cotton",
            5:"Coconut",
            6:"Papaya",
            7:"Orange",
            8:"Apple",
            9:"Muskmelon",
            10:"Watermelon",
            11:"Grapes",
            12:"Mango",
            13:"Banana",
            14:"Pomegranate",
            15:"Lentil",
            16:"Blackgram",
            17:"Mungbean",
            18:"Mothbeans",
            19:"Pigeonpeas",
            20:"Kidneybeans",
            21:"Chickpea",
            22:"Coffee"
        }

        result = crop_dict.get(int(prediction[0]), str(prediction[0]))

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)