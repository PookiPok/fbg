from lib2to3.pgen2.token import EQUAL
from operator import eq
from queue import Empty
from statistics import variance
from tokenize import Double
from typing_extensions import Self
import cv2
from cv2 import mean
import numpy as np
import mediapipe as mp
import math
from typing import List, Mapping, Optional, Tuple, Union
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

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int,image_height: int) -> Union[None, Tuple[int, int]]:

    # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))
  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x[0]*y[0]
        sumyy += x[1]*y[1]
        sumxy += x[0]*y[0]*x[1]*y[1]
    return sumxy/math.sqrt(sumxx*sumyy)

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
      try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      except:
        print("img have an issue")
    
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


  def createMeshPoints(self , landmark_list, image):
    idx_to_coordinates = {}
    image_rows, image_cols, _ = image.shape
    for idx, landmark in enumerate(landmark_list.landmark):
      if ((landmark.HasField('visibility') and
           landmark.visibility < _VISIBILITY_THRESHOLD) or
          (landmark.HasField('presence') and
          landmark.presence < _PRESENCE_THRESHOLD)):
        continue
      landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,image_cols, image_rows)
      if landmark_px:
        idx_to_coordinates[idx] = landmark_px
    return np.array(list(idx_to_coordinates.values()))

  def locateFaceMesh(self , img , draw = True):

    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    results = self.faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
      annotated_image = imgRGB.copy()
      """
      for face_landmarks in results.multi_face_landmarks:
        #print(face_landmarks)
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
      """
      #annotated_image = cv2.cvtColor(annotated_image , cv2.COLOR_RGB2BGR)
      #mesh_points = self.createMeshPoints(landmark_list=results.multi_face_landmarks[0],image=annotated_image)
      mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
      return annotated_image,mesh_points
    else:
      return []

  def getPivotNoseLocation(self, path):
    img = cv2.imread(path)
    result = self.locateFaceMesh(img , True)
    pivot_x = result[1][1][0]
    pivot_y = result[1][1][1]
    return pivot_x,pivot_y

  def createEmotionPicDirectory(self,accuracy=10):
    path = 'pictures/_pivot.jpg'
    pivot_x,pivot_y = self.getPivotNoseLocation(path)
    count = 0
    for subdir, dirs, files in os.walk('/home/ogi/Pictures/only_mesh'):
      for file in files:
        try:
          frame = cv2.imread(os.path.join(subdir, file))
          img, bboxs = self.locateFaces(frame)
          if bboxs:
            print("parsing img" + file)
            crop_img = bboxs[0][1]
            emotion_img = self.resizeImg(crop_img)
            new_img = cv2.resize(crop_img, (178,218))
            result_tmp = self.locateFaceMesh(new_img , True)
            if result_tmp:
              tmp_x = result_tmp[1][1][0]
              tmp_y = result_tmp[1][1][1]
              if (int(pivot_x-accuracy) < tmp_x < int(pivot_x+accuracy)) and (int(pivot_y-accuracy) < tmp_y < int(pivot_y+accuracy)):
                full_emotion = self.loadEmotion(emotion_img)
                if full_emotion['emotion_prediction'] > 0.85:
                  print("Taking pic:" + file + " With emotion:" + full_emotion['dominant_emotion'])
                  emotion = full_emotion['dominant_emotion']
                  path = 'pictures/'+ emotion
                  if os.path.exists(path) == False:
                    os.mkdir(path)
                  cv2.imwrite(path + '/' + file, new_img)
                  count = count + 1
        except BaseException as err:
          print(f"Unexpected {err=}, {type(err)=}") 
    print("Images taken:" + str(count))

  def createPivotPicture(self):
    frame = cv2.imread('pictures/100231.jpg')
    img, bboxs = self.locateFaces(frame)
    if bboxs:
      crop_img = bboxs[0][1]
      new_img = cv2.resize(crop_img, (178,218))
      cv2.imwrite('pictures/_pivot.jpg', new_img)

  def resizeImage(self, frame):
    img,bboxs = self.locateFaces(frame)
    if bboxs:
      crop_img = bboxs[0][1]
      try:
        new_img = cv2.resize(crop_img, (178,218))
      except:
        print("Error...")
    return new_img


  def detectVariance(self,real_mesh , mean_mesh, text='None'):
    var1 = np.var(real_mesh,axis=0)
    var2 = np.var(mean_mesh,axis=0)
    std1 = np.std(real_mesh,axis=0)
    std2 = np.std(mean_mesh,axis=0)
    mean1 = np.mean(real_mesh,axis=0)
    mean2 = np.mean(mean_mesh,axis=0)
    new_arry_std = np.abs(std1 - std2)
    new_arry_var = np.abs(var1 - var2)
    new_arry_mean = np.abs(mean1 - mean2) 
    std_total = np.std(np.array([[std1],[std2]]))
    var_total = np.std(np.array([[var1],[var2]]))
    arr1 = np.array(real_mesh)
    arr2 = np.array(mean_mesh)
    num_x = np.dot(arr1[0] , arr2[0])
    num_y = np.dot(arr1[1] , arr2[1])
    cos_sim_x = num_x / (np.linalg.norm(arr1[0])*np.linalg.norm(arr2[0]))
    cos_sim_y = num_y / (np.linalg.norm(arr1[1])*np.linalg.norm(arr2[1]))
    var_xy = np.var([[cos_sim_x,cos_sim_y]])
    std_xy = np.std([[cos_sim_x,cos_sim_y]])
    #cos_sim = np.dot(np.array[real_mesh], np.array[mean_mesh])/(np.linalg.norm(np.array[real_mesh])*np.linalg.norm(np.array[mean_mesh]))
    #cos_sim = cosine_similarity(real_mesh,mean_mesh)
    print(text + ": var_xy:"+ str(var_xy) + " std_xy:" + str(std_xy))
    #variance = np.var(new_Arry)
    #std = np.std(new_Arry)
    variance = np.var(np.var(np.array([[mean_mesh],[real_mesh]]),axis=0))
    std = np.std(np.std(np.array([[mean_mesh],[real_mesh]]),axis=0))
    #new_Arry = np.abs(mean_mesh - real_mesh)
    #variance = np.var(new_Arry)
    #std = np.std(new_Arry)
    #dist = np.linalg.norm(new_Arry)
    return variance,std

  def detectVarianceDiff(self,real_mesh , mean_mesh):
    variance = np.var(np.array([[mean_mesh],[real_mesh]]),axis=0)
    mean_var = np.mean(variance)
    std = np.std(np.array([[mean_mesh],[real_mesh]]),axis=0)
    mean_std = np.mean(std)
    return mean_var,mean_std

  def createPicSegment(self,std_min=1.5):
    #path = 'pictures/norm_pic'
    path_pic = 'pictures/happy'
    #pic1 = cv2.imread('pictures/norm_pic/186635.jpg') # take the first image
    pic1 = cv2.imread(path_pic + '/_pivot.jpg') # take the first image
    result_pic1 = self.locateFaceMesh(pic1 , True)
    all_mean_mesh = {}
    mean_mesh = result_pic1[1]
    dir_counter = 0
    all_mean_mesh[dir_counter] = mean_mesh
    for subdir, dirs, files in os.walk(path_pic):
      for file in files:
        try:
          found_std = False
          frame = cv2.imread(os.path.join(subdir, file))
          #cv2.imshow("real" , pic1)
          #cv2.imshow("test" , frame)
          #cv2.waitKey(0)
          result = self.locateFaceMesh(frame , True)
          if result:
            mesh_points = result[1]
            for i in all_mean_mesh:
              mean_mesh = all_mean_mesh[i]
              variance,std = self.detectVariance(mesh_points , mean_mesh)
              print("Variance:" + str(variance) + " Std:" + str(std) + " For image:" + file)
              if std < std_min:
                print("Variance:" + str(variance) + " Std:" + str(std) + " For image:" + file + " Place in bucket:" + str(i))
                mean_mesh = np.mean( np.array([ mean_mesh, mesh_points ]), axis=0 )
                path = path_pic + '/picseg_'+ str(i)
                if os.path.exists(path) == False:
                  os.mkdir(path_pic + '/picseg_'+ str(i))
                cv2.imwrite(path_pic + '/picseg_'+ str(i) + '/' + file, frame)
                all_mean_mesh[i] = mean_mesh
                found_std = True
                break 
            if found_std == False:
              path = path_pic + '/picseg_'+ str(i+1)
              if os.path.exists(path) == False:
                os.mkdir(path_pic + '/picseg_'+ str(i+1))
              cv2.imwrite(path_pic + '/picseg_'+ str(i+1) + '/' + file, frame)
              all_mean_mesh[i+1] = mesh_points

        except BaseException as err:
          print(f"Unexpected {err=}, {type(err)=}")
        #except:
          #print("Can't parse img:" + file)

  def createPicSegmentAll(self,std_min=2.0):
    #path_pic = 'pictures/norm_pic'
    path_pic = 'pictures/happy'
    #pic1 = cv2.imread('pictures/norm_pic/186501.jpg') # take the first image
    pic1 = cv2.imread(path_pic + '/_pivot.jpg') # take the first image
    result_pic1 = self.locateFaceMesh(pic1 , True)
    all_mean_mesh = {}
    mean_mesh = result_pic1[1]
    dir_counter = 0
    all_mean_mesh[dir_counter] = mean_mesh
    for subdir, dirs, files in os.walk(path_pic):
      for file in files:
        try:
          tuple_check = {}
          found_std = False
          frame = cv2.imread(os.path.join(subdir, file))
          #cv2.imshow("real" , pic1)
          #cv2.imshow("test" , frame)
          #cv2.waitKey(0)
          result = self.locateFaceMesh(frame , True)
          if result:
            mesh_points = result[1]
            for i in all_mean_mesh:
              mean_mesh = all_mean_mesh[i]
              variance,std = self.detectVariance(mesh_points , mean_mesh)
              print("Variance:" + str(variance) + " Std:" + str(std) + " For image:" + file)
              tuple_check[i] = std;
            
            min_location = min(tuple_check, key=tuple_check.get)
            std = tuple_check[min_location]
            if std < std_min:
              print(" Std:" + str(std) + " For image:" + file + " Place in bucket:" + str(min_location))
              mean_mesh = np.mean( np.array([ mean_mesh, mesh_points ]), axis=0 )
              path = path_pic + '/picseg_'+ str(min_location)
              if os.path.exists(path) == False:
                os.mkdir(path_pic + '/picseg_'+ str(min_location))
              cv2.imwrite(path_pic + '/picseg_'+ str(min_location) + '/' + file, frame)
              all_mean_mesh[min_location] = mean_mesh
              found_std = True
              #break 
          if found_std == False:
            min_location = i
            path = path_pic + '/picseg_'+ str(min_location+1)
            if os.path.exists(path) == False:
              os.mkdir(path_pic + '/picseg_'+ str(min_location+1))
            cv2.imwrite(path_pic + '/picseg_'+ str(min_location+1) + '/' + file, frame)
            all_mean_mesh[min_location+1] = mesh_points

        except BaseException as err:
          print(f"Unexpected {err=}, {type(err)=}")
        #except:
          #print("Can't parse img:" + file)

  def buildPictureArry(self, path_pic='pictures/happy'):
    heat_map = {k:None for k in range(0,478)}
    path = 'pictures/_pivot.jpg'
    pivot_x,pivot_y = self.getPivotNoseLocation(path)
    for subdir, dirs, files in os.walk(path_pic):
      for file in files:
        try:
          all_pic = []
          frame = cv2.imread(os.path.join(subdir, file))
          result = self.locateFaceMesh(frame , True)
          if result:
            mesh_nose_x = result[1][1][0]
            mesh_nose_y = result[1][1][1]
            div_x = pivot_x - mesh_nose_x
            div_y = pivot_y - mesh_nose_y
            mesh_points = result[1]
            for i in range(0,478):
              if heat_map[i]:
                tmp = mesh_points[i]
                tmp[0] = tmp[0] + div_x
                tmp[1] = tmp[1] + div_y
                all_pic = heat_map[i]
                all_pic.append(tmp)
              else:
                all_pic.append(mesh_points[i])
                all_pic[0][0] = all_pic[0][0] + div_x
                all_pic[0][1] = all_pic[0][1] + div_y 
                heat_map[i] = all_pic
              all_pic = []
        except BaseException as err:
          print(f"Unexpected {err=}, {type(err)=}")
    return heat_map

  def createPicHeatMapStdBase(self,accuracy=0.5):
    path_pic='pictures/happy'
    heat_map = self.buildPictureArry(path_pic=path_pic)
    np.savetxt
    std = None
    #pic1 = cv2.imread(path_pic + '/100858.jpg') # take the first image
    #result = self.locateFaceMesh(pic1 , False)
    #mesh = result[1]
    new_heat_map = []
    for i in heat_map:
      var_2d = np.var(np.array(heat_map[i]),axis=0)
      std_2d = np.std(np.array(heat_map[i]),axis=0)
      std = np.std(np.std(np.array(heat_map[i]),axis=0))
      var = np.var(heat_map[i])
      #std = np.std(heat_map[i])
      if std < accuracy:
          #x = int(mesh[i][0])
          #y = int(mesh[i][1])
          #cv2.circle(pic1,(x,y),1,(255,255,0),-1)
          new_heat_map.append(i)
      #heat_map[i] = std

    #img = cv2.resize(pic1 ,(720,720)) 
    #cv2.imshow("test" , img)
    #cv2.waitKey(0)
    return new_heat_map

  def createPicHeatMapMeanBase(self):
    full_pic_mean_file = 'stats/full_pic_mean.csv'
    full_heat_map_file = 'stats/full_heat_map.csv'
    full_pic_mean = []
    full_heat_map = []
    if os.path.exists(full_pic_mean_file) == True:
      full_pic_mean = np.array(np.loadtxt(full_pic_mean_file)).astype(float)
      full_heat_map = np.array(np.loadtxt(full_heat_map_file)).astype(float)
      return full_pic_mean,full_heat_map
    
    path_pic = 'pictures/happy'
    full_pic_arry_map = self.buildPictureArry(path_pic=path_pic)
    for i in full_pic_arry_map:
      line_array = np.array(full_pic_arry_map[i])
      xy_mean = np.mean(line_array,axis=0)
      full_pic_mean.append(xy_mean)
      sub_array = []
      for j in line_array:
        xy_sub = np.abs(j - xy_mean)
        sub_array.append(np.array(xy_sub))
      full_heat_map.append(np.array(np.std(sub_array,axis=0)))
    np.savetxt(full_pic_mean_file,full_pic_mean)
    np.savetxt(full_heat_map_file ,full_heat_map)
    return full_pic_mean,full_heat_map

  def createPicHeatMapStdBase(self):
    full_pic_mean_file = 'stats/full_pic_mean.csv'
    full_std_map_file = 'stats/full_std_map.csv'
    full_pic_mean = []
    full_std_map = []
    if os.path.exists(full_std_map_file) == True:
      full_pic_mean = np.array(np.loadtxt(full_pic_mean_file)).astype(float)
      full_std_map = np.array(np.loadtxt(full_std_map_file)).astype(float)
      return full_pic_mean,full_std_map
    
    path_pic = 'pictures/happy'
    full_pic_arry_map = self.buildPictureArry(path_pic=path_pic)
    for i in full_pic_arry_map:
      line_array = np.array(full_pic_arry_map[i])
      xy_mean = np.mean(line_array,axis=0)
      full_pic_mean.append(xy_mean)
      full_std_map.append(np.array(np.std(line_array,axis=0)))
    np.savetxt(full_pic_mean_file,full_pic_mean)
    np.savetxt(full_std_map_file ,full_std_map)
    return full_pic_mean,full_std_map

  def createPicHeatMapBaseOnDistance(self):
    #full_pic_mean_file = 'stats/full_dist_pic_mean.csv'
    full_pic_median_file = 'stats/full_dist_pic_median.csv'
    #full_std_mean_map_file = 'stats/full_dist_std_mean_map.csv'
    full_std_median_map_file = 'stats/full_dist_std_median_map.csv'
    full_std_mean_map_file = 'stats/full_dist_std_happy_median_map.csv'
    full_pic_mean_file = 'stats/full_dist_happy_pic_median.csv'
    full_pic_mean = []
    full_pic_median = []
    full_std_mean_map = []
    full_std_median_map = []
    if os.path.exists(full_std_median_map_file) == True:
      full_pic_mean = np.array(np.loadtxt(full_pic_mean_file)).astype(float)
      full_pic_median = np.array(np.loadtxt(full_pic_median_file)).astype(float)
      full_std_mean_map = np.array(np.loadtxt(full_std_mean_map_file)).astype(float)
      full_std_median_map = np.array(np.loadtxt(full_std_median_map_file)).astype(float)
      return full_pic_mean,full_pic_median,full_std_mean_map,full_std_median_map
    path_pic = 'pictures/neutral'
    full_pic_arry_map = self.buildPictureArry(path_pic=path_pic)
    for i in full_pic_arry_map:
      line_array = np.array(full_pic_arry_map[i])
      xy_mean = np.mean(line_array,axis=0)
      xy_median = np.median(line_array, axis=0)
      full_pic_mean.append(xy_mean)
      full_pic_median.append(xy_median)
      sub_array_mean = []
      sub_array_median = []
      for j in line_array:
        xy_distance_mean= math.dist([j[0],j[1]] ,[xy_mean[0],xy_mean[1]] )
        xy_distance_median= math.dist([j[0],j[1]] ,[xy_median[0],xy_median[1]] )
        sub_array_mean.append(xy_distance_mean)
        sub_array_median.append(xy_distance_median)
      full_std_mean_map.append(np.std(sub_array_mean))
      full_std_median_map.append(np.std(sub_array_median))
    np.savetxt(full_pic_mean_file,full_pic_mean)
    np.savetxt(full_pic_median_file,full_pic_median)
    np.savetxt(full_std_mean_map_file ,full_std_mean_map)
    np.savetxt(full_std_median_map_file ,full_std_median_map)
    return full_pic_mean,full_pic_median, full_std_mean_map,full_std_median_map

  def buildHeatMapMeshPoints(self, mesh_points , new_heat_map_keys, accuracy):
    new_mesh_heat_points = []
    for id,key in enumerate(new_heat_map_keys):
      if key <= accuracy:
        new_mesh_heat_points.append(mesh_points[id])
    return new_mesh_heat_points
   
  def extractAccuracyXYPoint(self,accuracy,full_heat_map):
    x_min,y_min = np.min(full_heat_map,axis=0)
    x_calc = x_min + x_min * accuracy
    y_calc = y_min + y_min * accuracy
    return x_calc,y_calc

  def extractAccuracyPoint(self,accuracy,full_heat_map):
    min = np.min(full_heat_map)
    calc = min + min * accuracy
    return calc


def pic_main():
  print("Starting program....")
  accuracy = 20
  detector = FaceDetector()
  #detector.createPicHeatMapStdBase(accuracy)
  #detector.createPivotPicture()
  #detector.createEmotionPicDirectory()
  #exit(0)
  #detector.createPicSegment()
  #detector.createPicSegmentAll()
  #full_pic_mean,full_heat_map = detector.createPicHeatMapMeanBase()
  #full_pic_mean,full_heat_map = detector.createPicHeatMapStdBase()
  full_pic_mean,full_pic_median,full_std_mean_map,full_std_median_map = detector.createPicHeatMapBaseOnDistance()
  mean_mesh_neutral = []
  mean_mesh_happy = []
  mean_mesh_neutral = detector.buildHeatMapMeshPoints(full_pic_median,full_std_median_map,accuracy)
  mean_mesh_happy = detector.buildHeatMapMeshPoints(full_pic_mean,full_std_mean_map,accuracy)

  path = 'pictures/_pivot.jpg'
  img = cv2.imread(path)
  pivot_x, pivot_y = detector.getPivotNoseLocation(path)
  #cv2.circle(img,(pivot_x,pivot_y),1,(0,0,0),-1) 
  #img = cv2.resize(img ,(720,720)) 
  #cv2.imshow("Final Result" , img)
  #cv2.waitKey(0)
  for subdir, dirs, files in os.walk('/home/ogi/Pictures/mesh_pic'):
    for file in files:
      new_file = os.path.join(subdir, file)
      img = cv2.imread(new_file)
      #img = detector.resizeImage(img)
      result = detector.locateFaceMesh(img , True)
      tmp_x = result[1][1][0]
      tmp_y = result[1][1][1]
      div_x = pivot_x - tmp_x
      div_y = pivot_y - tmp_y
      norm_mesh = result[1]
      for i in range(0,478):
        loc = norm_mesh[i]
        loc[0] = loc[0] + div_x
        loc[1] = loc[1] + div_y
      
      heat_map_real_points_with_happy = detector.buildHeatMapMeshPoints(norm_mesh,full_std_mean_map,accuracy)
      heat_map_real_points_with_neutral = detector.buildHeatMapMeshPoints(norm_mesh,full_std_median_map,accuracy)
      variance_h,std_h = detector.detectVariance(heat_map_real_points_with_happy , mean_mesh_happy,'happy')
      variance_n,std_n = detector.detectVariance(heat_map_real_points_with_neutral , mean_mesh_neutral, 'neutral')
      print("Variance To Happy:" + str(variance_h) + " Std:" + str(std_h) + " for image:" + file)
      print("Variance To Neutral:" + str(variance_n) + " Std:" + str(std_n) + " for image:" + file)
      emotion_img = detector.resizeImg(img)
      full_emotion = detector.loadEmotion(emotion_img)
      print(full_emotion['dominant_emotion'])
  #    f.writelines("Image name:"+file+ " std:" + str(std) + " var:" + str(variance))
  #    f.write('\n')
  #    for i in range(0,len(mean_mesh)):
  #      x = int(mean_mesh[i][0])
  #      y = int(mean_mesh[i][1])
  #      cv2.circle(img,(x,y),1,(0,0,0),-1) 
  #      x_real = int(heat_map_real_points[i][0])
  #      y_real = int(heat_map_real_points[i][1])
  #      cv2.circle(img,(x_real,y_real),1,(255,255,255),-1) 
      img = cv2.resize(img ,(720,720)) 
      cv2.imshow("Final Result" , img)
      cv2.waitKey(0)

  

if __name__ == "__main__":
  pic_main()