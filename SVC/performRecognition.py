import argparse as ap
import cv2
from skimage.feature import hog
from sklearn.externals import joblib
import numpy as np

parser = ap.ArgumentParser()
parser.add_argument("-classifier","--ClassifierPath", help="Path to Classifier File", required="True")
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

clf,pp = joblib.load(args["ClassifierPath"])

im = cv2.imread(args["image"])

im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret,im_th = cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)

img,ctrs,hier = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    cv2.rectangle(im,(rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
  
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
    nbr = clf.predict(roi_hog_fd)
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)      
    
    
cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
       
