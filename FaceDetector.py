from lib2to3.pgen2.token import EQUAL
from operator import eq
from queue import Empty
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import os
from deepface import DeepFace
from PIL import Image
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()


import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image


class FaceDetector():

  emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
  LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
  RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
  L_H_LEFT = [33]
  L_H_RIGHT = [133]
  R_H_LEFT = [362]
  R_H_RIGHT= [263]
  L_H_UP = [159]
  L_H_DOWN = [145]
  R_H_UP = [386]
  R_H_DOWN= [374]

  
  LEFT_EYE_IRIS = [469, 470, 471, 472]
  RIGHT_EYE_IRIS = [474, 475, 476, 477]
  eyesIdsList = [22,23,24,26,110,157,158,159,160,161,130,243]

  def __init__(self,minDetectionConfidence=0.5,actions=('age', 'gender', 'race', 'emotion')):
    self.minDetectionConfidence = minDetectionConfidence
    self.mpFaceDetection = mp.solutions.face_detection
    self.mpDraw = mp.solutions.drawing_utils
    self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)
    self.actions = list(actions)
    self.models = {}
    if 'emotion' in self.actions:
      self.models['emotion'] = DeepFace.build_model('Emotion')
    if 'age' in self.actions:
      self.models['age'] = DeepFace.build_model('Age')
    if 'gender' in self.actions:
      self.models['gender'] = DeepFace.build_model('Gender')
    if 'race' in self.actions:
      self.models['race'] = DeepFace.build_model('Race')
    # face mesh detector
    self.meshDetector = FaceMeshDetector(staticMode=True , maxFaces=1)
    self.initFaceMesh()

  def initFaceMesh(self, static_image_mode = False, max_num_faces = 1, refine_landmarks=True):
    self.face_mesh = mp.solutions.face_mesh
    self.faceMesh = self.face_mesh.FaceMesh(static_image_mode,
               max_num_faces,
               refine_landmarks,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5)
	

  def locateFaces(self,img,draw=True):
    
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    self.results = self.faceDetection.process(imgRGB)
    bboxs = []
    if self.results.detections:
        for id,detection in enumerate(self.results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
                    int(bboxC.width * iw ), int(bboxC.height * ih)
            self.drawRectangle(img,bbox)
            #cv2.putText(img, f'{int(detection.score[0]*100)}%',
            # (bbox[0] , bbox[1]-20) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
            # (255 , 0 , 255) , 2)
            x, y, w, h = bbox
            crop_img = img[y:(y+h), x:(x+w)]
            #cv2.imshow("Crop_img" , crop_img)
            bboxs.append([id, crop_img, bbox, detection.score])
    return img, bboxs


  def drawRectangle(self, img , bbox , l=30 , t=5,rt=1):
    x, y, w, h = bbox
    x1,y1 = x+w, y+h
    cv2.rectangle(img, bbox, (255,0,255), rt)

    #left up currner
    cv2.line(img, (x,y) , (x+l,y),(255,0,255),t) 
    cv2.line(img, (x,y) , (x,y+l),(255,0,255),t)

    #right up currner
    cv2.line(img, (x1,y) , (x1-l,y),(255,0,255),t) 
    cv2.line(img, (x1,y) , (x1,y+l),(255,0,255),t)

    #left down currner
    cv2.line(img, (x,y1) , (x+l,y1),(255,0,255),t) 
    cv2.line(img, (x,y1) , (x,y1-l),(255,0,255),t)

    #right down currner
    cv2.line(img, (x1,y1) , (x1-l,y1),(255,0,255),t) 
    cv2.line(img, (x1,y1) , (x1,y1-l),(255,0,255),t)


  def loadEmotion(self, img):
    self.resp_obj = {}
    self.emotion_predictions = self.models['emotion'].predict(img)[0,:]
    self.sum_of_predictions = self.emotion_predictions.sum()
    self.resp_obj["emotion"] = {}

    for i in range(0, len(self.emotion_labels)):
      self.emotion_label = self.emotion_labels[i]
      self.emotion_prediction = 100 * self.emotion_predictions[i] / self.sum_of_predictions
      self.resp_obj["emotion"][self.emotion_label] = self.emotion_prediction

    self.resp_obj["dominant_emotion"] = self.emotion_labels[np.argmax(self.emotion_predictions)]
    return self.resp_obj["dominant_emotion"] 

  def resizeImg(self, img, target_size=(48, 48), grayscale=True):

    if grayscale == True:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.shape[0] > 0 and img.shape[1] > 0:
      factor_0 = target_size[0] / img.shape[0]
      factor_1 = target_size[1] / img.shape[1]
      factor = min(factor_0, factor_1)

      dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
      img = cv2.resize(img, dsize)
      
      # Then pad the other side to the target size by adding black pixels
      diff_0 = target_size[0] - img.shape[0]
      diff_1 = target_size[1] - img.shape[1]
      if grayscale == False:
        # Put the base image in the middle of the padded image
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
      else:
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#------------------------------------------
    
    #double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
      img = cv2.resize(img, target_size)
	#---------------------------------------------------
    #normalizing the image pixels
    img_pixels = image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]
    return img_pixels


  def locateEyeFaceMesh(self , img , draw = True):

    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    results = self.faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
      mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
      cv2.polylines(img, [mesh_points[self.LEFT_EYE]] , True, (0,255,0), 1, cv2.LINE_AA)
      cv2.polylines(img, [mesh_points[self.RIGHT_EYE]] , True, (0,255,0), 1, cv2.LINE_AA)
      (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[self.LEFT_EYE_IRIS])
      (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[self.RIGHT_EYE_IRIS])
      center_left = np.array([l_cx, l_cy],dtype=np.int32)
      center_right = np.array([r_cx, r_cy],dtype=np.int32)
      cv2.circle(img, center_left,int(l_radius), (0,0,255), 1, cv2.LINE_AA)
      cv2.circle(img, center_right,int(r_radius), (0,0,255), 1, cv2.LINE_AA)
      cv2.circle(img, mesh_points[self.R_H_RIGHT][0],3, (255,255,255), -1, cv2.LINE_AA)
      cv2.circle(img, mesh_points[self.R_H_LEFT][0],3, (0,255,255), -1, cv2.LINE_AA)
      iris_position, ratio = self.iris_position(center_right, mesh_points[self.R_H_RIGHT][0], mesh_points[self.R_H_LEFT][0])
      isclose = self.isEyesClose(mesh_points)
      cv2.putText(img, f' Location:{str(iris_position)}',
                (0 , 50) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
                (0, 0 , 0) , 2)
      cv2.putText(img, f' Eye Close:{str(isclose)}',
                (0 , 100) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
                (0, 0 , 0) , 2)
      print(iris_position + "----" + str(ratio) + "-----" + str(isclose))

    return img

  def locateFaceMesh(self , img , draw = True):

    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    results = self.faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
      mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
      print(mesh_points)
    return img

  def isEyesClose(self, mesh_points,diss = 30):

    eyesVertical,_ = self.meshDetector.findDistance(mesh_points[self.L_H_UP][0], mesh_points[self.L_H_DOWN][0])
    eyeHorizontal,_ = self.meshDetector.findDistance(mesh_points[self.L_H_LEFT][0], mesh_points[self.L_H_RIGHT][0])
    eyeclose = int((eyesVertical/eyeHorizontal) * 100)
    if eyeclose < diss:
      return True
    return False

  def euclidean_distance(self, point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

  def iris_position(self, iris_center, right_point, left_point):
    center_to_right_dist = self.euclidean_distance(iris_center, right_point)
    total_distance = self.euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position = ""
    if ratio < 0.42:
      iris_position = "right"
    elif ratio>0.42 and ratio<=0.57:
      iris_position = "center"
    else:
      iris_position = "left"
    return iris_position, ratio


  def plotImage(self, num):
    im_normed = np.random.rand(6, 6)
    ax.imshow(im_normed)
    ax.set_axis_off()
   


def main():

  cap = cv2.VideoCapture('video/2.mp4')
  #cap = cv2.VideoCapture(0)
  fps = cap.get(cv2.CAP_PROP_FPS)
  fpsPerSec = fps * 1 # 5 sec 
  print("FPS:" + str(fpsPerSec))

  pTime = 0
  detector = FaceDetector()
  count = 1
  while True:
    success, img = cap.read()
    if count%fpsPerSec==0:
      img, bboxs = detector.locateFaces(img)
      meshimg = detector.locateEyeFaceMesh(img , True)
      if bboxs:
        crop_img = bboxs[0][1]
        new_img = detector.resizeImg(crop_img)
        dominant_emotion = detector.loadEmotion(new_img)
        cv2.putText(meshimg, f'{str(dominant_emotion)}',
                (bboxs[0][2][0] , bboxs[0][2][1]-20) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
                (255, 0 , 255) , 2)
        cv2.imshow("Eye", meshimg)
        print(dominant_emotion)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == 27:
      break
    count+=1
  cap.release()
  cv2.destroyAllWindows()


def pic_main():
  detector = FaceDetector()
  for subdir, dirs, files in os.walk('pictures/happy'):
    for file in files:
      frame = cv2.imread(os.path.join(subdir, file))
      img, bboxs = detector.locateFaces(frame)
      #meshimg = detector.locateFaceMesh(img , True)
      if bboxs:
        crop_img = bboxs[0][1]
        new_img = detector.resizeImg(crop_img)
        meshimg = detector.locateFaceMesh(crop_img , True)
        dominant_emotion = detector.loadEmotion(new_img)
        cv2.putText(meshimg, f'{str(dominant_emotion)}',
                (bboxs[0][2][0] , bboxs[0][2][1]-20) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
                (255, 0 , 255) , 2)
        cv2.imshow("Eye", meshimg)
        print(dominant_emotion)
        cv2.imshow("Image" , img) 
        key = cv2.waitKey(0)




if __name__ == "__main__":
  #main()
  pic_main()