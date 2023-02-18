from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

model=pickle.load(open('finalized_model.sav','rb'))


@app.route('/')
def hello_world():
    return render_template("heart_stroke.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    scaler = StandardScaler()
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df.dropna()
    X = df.drop("stroke",axis = 1)
    X = X.drop("id",axis=1)
    X = X.drop("Residence_type",axis=1)
    cat_value = X.select_dtypes(exclude=np.number)
    features = [x for x in request.form.values()]
    print(features)
    featuresnp = np.array(features)
    print(featuresnp)
    dfty1 = pd.DataFrame(featuresnp).T
    dfty1.columns = X.columns
    print(dfty1)
    dfty1[["age", "hypertension", "bmi", "avg_glucose_level", "heart_disease"]] = dfty1[["age", "hypertension", "bmi", "avg_glucose_level", "heart_disease"]].apply(pd.to_numeric)
    X = pd.concat([X, dfty1], ignore_index=True)
    X = pd.get_dummies(X,columns=cat_value.columns)
    xColumns = X.columns
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X,columns=xColumns)
    trylast = X.tail(1)
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    prediction=loaded_model.predict_proba(trylast)
    print(prediction)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    if output>str(0.5):
        return render_template('heart_stroke.html',pred='Your Heart is in Danger.\nProbability of stroke occuring is {}'.format(output))
    else:
        return render_template('heart_stroke.html',pred='Your Heart is safe.\n Probability of stroke occuring is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
