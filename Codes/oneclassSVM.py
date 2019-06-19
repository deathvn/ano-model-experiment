import numpy as np
import cv2
import os
from sklearn.svm import OneClassSVM
from load_gt import load_labels, load_psnr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn import metrics
import pickle
import imutils

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True
print("[+] Setup model")
model = VGG16(weights='imagenet', include_top=False)


def extract_vgg16_features(path):
    img = image.load_img(path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data).flatten()
    return features


def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()
    
def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
 
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()
    
def load(container, path):
    #path : path to frames folder
    #container : train <or> test
    path_list = []
    folders = os.listdir(path)
    folders.sort()
    for fol in folders:
        imgs = os.listdir(path + '/' + fol)
        imgs.sort(key = lambda imgs:int(imgs.split('.')[0]))
        for img in imgs:
            img_path = path + '/' + fol + '/' + img
            print ("Extracting", img_path)
            path_list.append(img_path)
            image_dat = cv2.imread(img_path)
            #feature = image_to_feature_vector(image_dat)
            feature = extract_color_histogram(image_dat)
            #feature = extract_vgg16_features(img_path)
            container.append(feature)
    return path_list

def get_scores(path):
    scores = np.load(path)
    scores = scores.reshape(-1, 1)
    return scores
    
def get_psnr(path):
    psnr = []
    psnr_record = np.load(path, allow_pickle=True)
    for k in psnr_record:
        for val in k[4:]:
            psnr.append(val)
    psnr = np.array(psnr).reshape(-1, 1)
    return psnr

model_name = 'psnr_feat.sav'

###Training pharse
#train = []
#load(train, 'temp_image/train/frames')
#train = get_scores('scores_train_feat.npy')
train = get_psnr('psnr_train_feat.npy')
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(train)
pickle.dump(clf, open(model_name, 'wb'))
y_pred_train = clf.predict(train)
print ("error train:", y_pred_train[y_pred_train == -1].size)



###Testing pharse
_, _, labels = load_labels()
#test = []
#test_list = load(test, 'temp_image/test/frames')
#test = get_scores('scores_test_feat.npy')
test = get_psnr('psnr_test_feat.npy')
#print ("test", test)
clf = pickle.load(open(model_name, 'rb'))
y_pred_test = clf.predict(test)
d = clf.decision_function(test)
d_MAX = max(d)
d = [ d_MAX-i for i in d]

y_pred_test[y_pred_test==1] = 0
y_pred_test[y_pred_test==-1] = 1

normal = 0
abnormal = 0
el = 0
for i in y_pred_test:
    if i==0:
        abnormal+=1
    elif i==1:
        normal+=1
    else:
        el+=1
print (abnormal)
print (normal)
print (el)
  
print (y_pred_test)


print ("d length", len(d))
print ("label length", len(labels))
fpr, tpr, thresholds = metrics.roc_curve(labels, d, pos_label=1)
auc = metrics.auc(fpr, tpr)
print ("AUC", auc)


acc = accuracy_score(labels, y_pred_test)
f1 = f1_score(labels, y_pred_test)
precision = precision_score(labels, y_pred_test)
recall = recall_score(labels, y_pred_test)
print ('acc', acc)
print ('f1', f1)
print ('precision', precision)
print ('recall', recall)
