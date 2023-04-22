from lib2to3.pgen2.token import EQUAL
from operator import eq
import cv2
import numpy as np
import mediapipe as mp
import os
from deepface import DeepFace


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

  def __init__(self,minDetectionConfidence=0.5,actions=('age', 'gender', 'race', 'emotion')):
    self.minDetectionConfidence = minDetectionConfidence
    self.mpFaceDetection = mp.solutions.face_detection
    self.mpDraw = mp.solutions.drawing_utils
    self.faceDetection = self.mpFaceDetection.FaceDetection()
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


  def locateFaces(self,img,draw=True):
    
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    self.results = self.faceDetection.process(imgRGB)
    bboxs = []
    if self.results.detections:
        for id,detection in enumerate(self.results.detections):
            bbox = detection.location_data.relative_bounding_box
            h, w, c = imgRGB.shape
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            xmax = int((bbox.xmin + bbox.width) * w)
            ymax = int((bbox.ymin + bbox.height) * h)
            #bboxC = detection.location_data.relative_bounding_box
            #ih, iw, ic = img.shape
            #if (bboxC.xmin < 0):
            #  bboxC.xmin = 0
            #if (bboxC.ymin < 0):
            #  bboxC.ymin = 0
            #bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
            #        int(bboxC.width * iw ), int(bboxC.height * ih)
            #self.drawRectangle(img,bbox)
            #cv2.putText(img, f'{int(detection.score[0]*100)}%',
            # (bbox[0] , bbox[1]-20) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
            # (255 , 0 , 255) , 2)
            #x, y, w, h = bbox
            #crop_img = img[y:(y+h), x:(x+w)]
            bbox = xmin, ymin, xmax, ymax
            crop_img = img[ymin:(ymin+ymax), xmin:(xmin+xmax)]
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

def main():

  cap = cv2.VideoCapture('video/5.mp4')
  #cap = cv2.VideoCapture(0)
  fps = cap.get(cv2.CAP_PROP_FPS)
  fpsPerSec = round(fps, 0)
  fpsWaitTime = fpsPerSec * 1
  print("FPS:" + str(fpsPerSec))

  pTime = 0
  detector = FaceDetector()
  count = 1
  while True:
    success, img = cap.read()
    if count%fpsWaitTime == 0:
      img, bboxs = detector.locateFaces(img)
      #meshimg = detector.locateEyeFaceMesh(img , True)
      if bboxs:
        crop_img = bboxs[0][1]
        new_img = detector.resizeImg(crop_img)
        emotions = detector.loadEmotion(new_img)
        dominant_emotion = emotions['dominant_emotion']
        cv2.putText(img, f'{str(dominant_emotion)}',
                (bboxs[0][2][0] , bboxs[0][2][1]-20) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
                (255, 0 , 255) , 2)
        cv2.imshow("Emotion Image:", img)
        print(emotions)
        path = 'pictures/'+ dominant_emotion
        #if os.path.exists(path) == False:
            #os.mkdir(path)
        #cv2.imwrite(path + '/' + str(count) + ".jpg", crop_img)

    cv2.imshow("Image", img)

    key = cv2.waitKey(int(fpsPerSec))
    if key == 27:
      break
    count+=1
  cap.release()
  cv2.destroyAllWindows()


def pic_main():
  detector = FaceDetector()
  for subdir, dirs, files in os.walk('pictures/angry'):
    for file in files:
      frame = cv2.imread(os.path.join(subdir, file))
      img, bboxs = detector.locateFaces(frame)
      if bboxs:
        new_img = detector.resizeImg(frame)
        emotions = detector.loadEmotion(new_img)
        dominant_emotion = emotions['dominant_emotion']
        cv2.putText(img, f'{str(dominant_emotion)}',
                (bboxs[0][2][0] , bboxs[0][2][1]-20) , cv2.FONT_HERSHEY_SIMPLEX, 2 , 
                (255, 0 , 255) , 2)
        cv2.imshow("Emotion", img)
        print(dominant_emotion)
        key = cv2.waitKey(0)

def gpt_main():
   detector = FaceDetector()
   count = 1
   video_capture = cv2.VideoCapture('video/6.mp4')
   fps = video_capture.get(cv2.CAP_PROP_FPS)
   fpsPerSec = round(fps, 0)
   fpsWaitTime = fpsPerSec * 1
   while True:
    ret, img = video_capture.read()

    if not ret:
        break
      
    if count%fpsWaitTime == 0:
      # Convert the frame to RGB format
      frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # Run face detection on the frame
      results = detector.faceDetection.process(frame)

      if results.detections:
      # Draw the detected faces on the frame
          for detection in results.detections:
              bbox = detection.location_data.relative_bounding_box
              h, w, c = frame.shape
              xmin = int(bbox.xmin * w)
              ymin = int(bbox.ymin * h)
              xmax = int(bbox.width * w)
              ymax = int(bbox.height * h)
              cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
              crop_img = img[ymin:(ymin+ymax), xmin:(xmin+xmax)]
              if crop_img.size > 0:
                new_img = detector.resizeImg(crop_img)
                emotions = detector.loadEmotion(new_img)
                dominant_emotion = emotions['dominant_emotion']
                print(emotions)
                path = 'pictures/'+ dominant_emotion
                if os.path.exists(path) == False:
                  os.mkdir(path)
                cv2.imwrite(path + '/' + str(count) + ".jpg", crop_img)
                #cv2.imshow('Face Detection', crop_img)

            

          # Display the processed frame
          #cv2.imshow('Face Detection', crop_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count+=1




if __name__ == "__main__":
  #main()
  gpt_main()
  #pic_main()