import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy import fftpack
#import death25



# 音声データを読み込む
speakers = {'fly' : 0, 'growl' : 1, 'whistle' : 2, 'pig' : 3, 'clean' : 4}

# 特徴量を返す
def get_feat(file_name):
    a, sr = librosa.load(file_name)
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

data_X, data_y = get_data('voices2')
# get_feat('sample/hi.wav')
# get_feat('sample/lo.wav')
test_X, test_y = get_data('output3')
# 教師データとテストデータに分ける
#train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, random_state=11813)
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
    
   
  

def predict2(X):
    result = clf.predict(X.T)
    
    
    return np.where(np.bincount(result)==np.unique(np.bincount(result))[-2])[0]
    
ZH = ''
ok_count = 0
score = 0
for X, y in zip(test_X, test_y):
    actual = predict(X)
    death = predict2(X)
    
    expected = y[0]
    file_name = y[1]
    ok_count += 1 if actual == expected else 0
    result = 'o' if actual == expected else 'x'
    
    JP = None
    
    if actual == 0:
       JP ='flyscream'
       score += 3
    elif actual == 1:
       JP = 'growl'
       score += 3
    elif actual == 2:
       JP = 'whistle'
       score += 3
    else:
       JP ='pig'
       score += 3
       
       
    HP = ''
    for INP in death:
       if INP == 0:
          HP += 'flyscream'

       elif INP == 1:
          HP += 'growl'
    
       elif INP == 2:
          HP += 'whistle'
    
       elif INP == 3:
          HP += 'pig'

       else :
          HP += 'None'
          
          
       #for ZZ in str(actual):
         #if ZZ == '0':
           #ZH += '(flyscream) '
            
         #elif ZZ == '1':
            #ZH += '(growl) '
                
         #elif ZZ == '2':
            #ZH += '(whistle) '
                
         #elif ZZ == '3':
            #ZH += '(pig) '
            
         #else :
            #ZH += '(None) '
    print( HP +'型の'+JP +'が検出されました' )
    

    

    #print("{} file: {}, actual: {}, expected: {}".format(result, file_name, actual, expected))
    

#print("{}/{}".format(ok_count, len(test_X)))

print("デスボイス加点は"+str(score)+"点です")

#import c2

#c2inst = c2.HHV()