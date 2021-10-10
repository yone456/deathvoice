import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy import fftpack
import wave
import struct
import math
from scipy import fromstring, int16
#from scipy.io.wavfile import write
import shutil

# 一応既に同じ名前のディレクトリがないか確認。
file = os.path.exists("output")
#file2 = os.path.exists("output2")
print(file)
#print(file2)

if file == False:
    #保存先のディレクトリの作成
    os.mkdir("output")
    #os.mkdir("output2")

def cut_wav(filename,time):  # WAVファイルを刈り奪る　形をしてるだろ？ 
    # timeの単位は[sec]

    # ファイルを読み出し
    wavf = filename + '.wav'
    wr = wave.open(wavf, 'r')

    # waveファイルが持つ性質を取得
    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate()
    fn = wr.getnframes()
    total_time = 1.0 * fn / fr
    integer = math.floor(total_time) # 小数点以下切り捨て
    t = int(time)  # 秒数[sec]
    frames = int(ch * fr * t)
    num_cut = int(integer//t)

    #　確認用
    print("Channel: ", ch)
    print("Sample width: ", width)
    print("Frame Rate: ", fr)
    print("Frame num: ", fn)
    print("Params: ", wr.getparams())
    print("Total time: ", total_time)
    print("Total time(integer)",integer)
    print("Time: ", t) 
    print("Frames: ", frames) 
    print("Number of cut: ",num_cut)

    # waveの実データを取得し、数値化
    data = wr.readframes(wr.getnframes())
    wr.close()
    X = fromstring(data, dtype=int16)
    print(X)


    for i in range(num_cut):
        print(i+1)
        # 出力データを生成
        outf = 'output/' +'clean' + '_' + ' '+ '(' + str(i+1)+ ')'+'.wav' 
        start_cut = i*frames
        end_cut = i*frames + frames
        print(start_cut)
        print(end_cut)
        Y = X[start_cut:end_cut]
        outd = struct.pack("h" * len(Y), *Y)

        # 書き出し
        ww = wave.open(outf, 'w')
        ww.setnchannels(ch)
        ww.setsampwidth(width)
        ww.setframerate(fr)
        ww.writeframes(outd)
        ww.close()
#shutil.copytree('output', 'output2')
print("input filename = ")
f_name = input()
print("cut time = ")
cut_time = input()
cut_wav(f_name,cut_time)
shutil.copytree('output', 'output2') 





# 音声データを読み込む
speakers = {'deathvoice' : 0, 'clean' : 1}

# 特徴量を返す
def get_feat(file_name):
    a, sr = librosa.load(file_name)
    y = librosa.load(file_name)
   # fft_wave = fftpack.rfft(a, n=sr)
    #fft_freq = fftpack.rfftfreq(n=sr, d=1/sr)
   
# y = librosa.amplitude_to_db(fft_wave, ref=np.max)
    y = librosa.feature.mfcc(y=a, sr=sr)
#   plt.plot(fft_freq, y)
#   plt.show()
    return y

# 特徴量と分類のラベル済みのラベルの組を返す
def get_data(dir_name):
    data_X = []
    data_y = []
    for file_name in sorted(os.listdir(path=dir_name)):
        print("read: {}".format(file_name))
        speaker = file_name[0:file_name.index('_')]
        data_X.append(get_feat(os.path.join(dir_name, file_name)))
        data_y.append((speakers[speaker], file_name))
    return (data_X,data_y)

data_X, data_y = get_data('voices4')
# get_feat('sample/hi.wav')
# get_feat('sample/lo.wav')
test_X, test_y = get_data('output')
# 教師データとテストデータに分ける
#train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, random_state=813)
#print("{} -> {}, {}".format(len(data_X), len(train_X), len(test_X)))


# 各時間に対応する成分をそれぞれ1つの特徴量として分割する
def split_feat(data_X, data_y):
    data_X2 = []
    data_y2 = []
    for X, y in zip(data_X, data_y):
        X2 = X.T
        y2 = np.array([y[0]] * X.shape[1])
        data_X2.append(X2)
        data_y2.append(y2)
    data_X2 = np.concatenate(data_X2)
    data_y2 = np.concatenate(data_y2)
    
    return (data_X2, data_y2)

train_X2, train_y2 = split_feat(data_X, data_y)

# clf = svm.SVC(gamma=0.0001, C=1)
clf = svm.SVC(gamma=0.0000001, C=10)
clf.fit(train_X2, train_y2)

def predict(X):
    result = clf.predict(X.T)
    return np.argmax(np.bincount(result))

ok_count = 0
os.mkdir('output3')
for X, y in zip(test_X, test_y):
    actual = predict(X)
    expected = y[0]
    file_name = y[1]
    file_name2 = 'output2/' + file_name
    ok_count += 1 if actual == expected else 0
    result = 'o' if actual == expected else 'x'
    #wavz = file_name 
    #wi = wave.open(wavz, 'r')
    #ch = wi.getnchannels()
    #width = wi.getsampwidth()
    #fr = wi.getframerate()
    if actual == 0:
        shutil.move(file_name2, 'output3')
         #write(file_name, 44100, X)
         #wavz = test_y
         #wi = wave.open('output', 'r')
         #ch = wi.getnchannels()
         #width = wi.getsampwidth()
         #fr = wi.getframerate()
         #out = 'output2/' + file_name
         #Z = wave.open(out, 'w')
         #Z.setnchannels(ch)
         #Z.setsampwidth(width)
         #Z.setframerate(fr)
         #Z.close()
        
    #print("{} file: {}, actual: {}, expected: {}".format(result, file_name, actual, expected))
    

#print("{}/{}".format(ok_count, len(test_X)))


