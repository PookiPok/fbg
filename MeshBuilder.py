from lib2to3.pgen2.token import EQUAL
from operator import eq
from queue import Empty
from tokenize import Double
from typing_extensions import Self
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
    self.mp_drawing_styles = mp.solutions.drawing_styles
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
    self.initFaceMesh(True)

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
    self.resp_obj["emotion_prediction"] = np.max(self.emotion_predictions)
    return self.resp_obj 

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

  def locateFaceMesh(self , img , draw = True):

    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    results = self.faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
      annotated_image = imgRGB.copy()
      for face_landmarks in results.multi_face_landmarks:
        #print(face_landmarks)
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        self.mpDraw.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        self.mpDraw.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_contours_style())
        self.mpDraw.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
      return annotated_image,mesh_points,face_landmarks
    else:
      return None,None,None

def createEmotionPicDirectory(self):

  for subdir, dirs, files in os.walk('/home/ogi/Pictures/mesh_pic_20'):
    for file in files:
      frame = cv2.imread(os.path.join(subdir, file))
      img, bboxs = self.locateFaces(frame)
      if bboxs:
        crop_img = bboxs[0][1]
        emotion_img = self.resizeImg(crop_img)
        new_img = cv2.resize(crop_img, (256,256))
        full_emotion = self.loadEmotion(emotion_img)
        if full_emotion['emotion_prediction'] > 0.90:
          print(full_emotion['dominant_emotion'])
          emotion = full_emotion['dominant_emotion']
          cv2.imwrite('pictures/'+emotion+'/' + file, new_img)

def pic_main():
  detector = FaceDetector()
  #createEmotionPicDirectory(detector)
  all_mesh = {}
  count = 0
  for subdir, dirs, files in os.walk('pictures/happy'):
    for file in files:
      frame = cv2.imread(os.path.join(subdir, file))
      result = detector.locateFaceMesh(frame , True)
      if result:
        meshimg = result[0]
        mesh_points = result[1]
        m_p = np.array([np.multiply([p.x, p.y, p.z], [1, 1, 1]).astype(float) for p in result[2].landmark])
        all_mesh[count] = mesh_points
        count = count + 1
        #cv2.imshow("Image" , meshimg) 
        #key = cv2.waitKey(0)
  new_mesh = all_mesh[0]
  for place in all_mesh:
    if place == len(all_mesh)-1:
      break 
    new_mesh = np.mean( np.array([ new_mesh, all_mesh[place+1] ]), axis=0 )
    
    #print(new_mesh)
  mean_mesh = np.ceil(new_mesh)
  #img = cv2.imread('pictures/happy/_pivot.jpg')
  img = cv2.imread('pictures/happy/200426.jpg')
  result = detector.locateFaceMesh(img , True)
  real_mesh = result[1] 
  for i in range(0,478):
    x = int(mean_mesh[i][0])
    y = int(mean_mesh[i][1])
    x1 = int(real_mesh[i][0])
    y2 = int(real_mesh[i][1])
    cv2.circle(img,(x,y),1,(100,100,0),-1)
    cv2.circle(img,(x1,y2),1,(255,255,0),-1)
    
  img = cv2.resize(img ,(720,720))
  cv2.imshow("Final Result" , img)
  cv2.waitKey(0)

  

if __name__ == "__main__":
  pic_main()