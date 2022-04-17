import librosa as lr
import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,render_template,jsonify
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='static/templates')

@app.route('/get_emotion', methods=['POST'])
def get_emotion():
    audio_data = request.files['audio_data']

    audio_wave_data,sampling_rate = lr.load(audio_data)
    with open('svc_model.pickle' , 'rb') as f_svc:
        svc = pickle.load(f_svc)
    with open('scaler_model.pickle' , 'rb') as f_sc:
        sc = pickle.load(f_sc)

    x = np.arange(0,148)      #final extracted feature will be stored here so total=128+20
    row_data = np.array([])     #store each rowdata
    mfcc = lr.feature.mfcc(audio_wave_data,sr=sampling_rate).T      #generate mfcc data and transpose
    mel_spec = lr.feature.melspectrogram(audio_wave_data,sr=sampling_rate)     #generate mfcc data
    mel_spec_db = lr.power_to_db(mel_spec, ref=np.max).T        #to dB and transpose
    mfcc_mean = np.mean(mfcc,axis = 0)          #calculate mean of mfcc along time axis
    row_data = np.hstack((row_data,mfcc_mean))
    mel_spec_mean = np.mean(mel_spec_db,axis = 0)      #calculate mean of mel_spec along time axis
    row_data = np.hstack((row_data,mel_spec_mean))    
    x = np.vstack((x,row_data))

    x_data = pd.DataFrame(x)
    x_data.drop(0,axis=0,inplace=True)
    x_data.reset_index(drop=True,inplace=True)

    x_data = sc.transform(x_data)

    y_pred = pd.Series(svc.predict(x_data))
    y_pred = y_pred.map({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
    return jsonify(y_pred[0])

@app.route('/', methods=['get'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)