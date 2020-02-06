# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:34:57 2020

@author: omen
"""

from tkinter import *
from tkinter import filedialog
 
from PIL import Image,ImageTk
import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import MinMaxScaler
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

pd.set_option('max_info_columns', 999)
pd.options.display.max_rows = 200
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import StandardScaler



import utils


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rms=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(DIR):

    features = pd.Series(index=columns(), dtype=np.float32)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        
        
        x, sr = librosa.load(DIR, sr=None, mono=True)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rms(S=stft)
        feature_stats('rms', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}'.format(repr(e)))

    return features

def generate_matrix(DIR):
    X = compute_features(DIR)
    Y = compute_features('predict_samples/bourrage.mp3')
    X = X.to_frame().T
    X2 = Y.to_frame().T

    X = pd.concat([X,X2])
    print(X.head())
    X = StandardScaler().fit_transform(X)
    np.save("03.npy", X)
    
def load_matrix(loaded_model):
    Xtesst1 =np.load("03.npy",allow_pickle=True)
    Xtest2 = MinMaxScaler().fit_transform(Xtesst1)
    y = loaded_model.predict(Xtest2)
    return y
def show_plot(y_X):
    dict_genres = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 
               'Instrumental':4,'International':5, 'Pop' :6, 'Rock': 7  }
    key_list = list(dict_genres.keys()) 
    val_list = list(dict_genres.values()) 
    j = 0
    a = max(y_X[0])
    print(a)
    for i in range(len(y_X[0])-1):
        if y_X[0][i] == a:
            j = i 
            df = pd.DataFrame({'genre':['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 
               'Instrumental','International', 'Pop' , 'Rock'], '%':y_X[0]*100})
    ax = df.plot.bar(x='genre', y='%', rot=0)
    ax.get_figure().savefig('4.png')
    
    
    
#------------------------------------------------------------------------------------------------------------------------

def popup():
    win = Toplevel()
    win.wm_title("Plot")
    image = PhotoImage(file="4.png")
    label = Label(win,image=image)
    label.pack()
    win.mainloop()


class Interface(Frame):

    def __init__(self, fenetre, **kwargs):
        Frame.__init__(self, fenetre, width=1200, height=800, **kwargs)
        self.pack(fill=BOTH)
        
        self.message = Label(self, text="UPLOAD AN MP3 TRACK")
        self.message.pack()
        
        self.bouton_Generate = Button(self, text="Generate", fg="red",command=self.Generate)
        self.bouton_Generate.pack(side="right")
       
        self.bouton_showPlot = Button(self, text="showPlot", fg="red",command=popup)
        self.bouton_showPlot.pack(side="left")
        
        self.bouton_browse = Button(self, text="Browse",command=self.load_file)
        self.bouton_browse.pack(side="right")
        
        self.bouton_clear = Button(self, text="Clear",command=self.clear)
        self.bouton_clear.pack(side="left")
    
    
    def load_file(self):
        
        self.file_name = filedialog.askopenfilename(filetypes = [("mp3 files","*.mp3"),("all files","*.*")])
    
    def Generate(self):
        d = self.file_name
       
        generate_matrix(d)
        model = loaded_model = tf.keras.models.load_model('final_model.h5')
        Y = load_matrix(model)
        show_plot(Y)
        print("end generating")
        
    def clear(self):
        self.label.destroyè()
        
        
    
        

        
        
fenetre = Tk()
fenetre.geometry("200x100")
interface = Interface(fenetre)
interface.mainloop()
interface.destroy()