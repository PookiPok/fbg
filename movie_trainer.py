import cv2
import os
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import argparse
import joblib
import utils


class MovieTrainer():
    def __init__(self,location=os.getcwd() + "\\",image_mode=True, max_faces=2, model='golden_trained.pkl'):
      self.loc = location
      self.smile_icon = cv2.imread(self.loc + 'logo\\happy_logo.png',cv2.IMREAD_UNCHANGED)
      self.smile_icon = cv2.resize(self.smile_icon, (0,0), None, 0.4,0.4)
      self.not_smile_icon = cv2.imread(self.loc + 'logo\\not_happy_logo.png', cv2.IMREAD_UNCHANGED)
      self.not_smile_icon = cv2.resize(self.not_smile_icon, (0,0), None, 0.4,0.4)
      self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=image_mode, max_num_faces=max_faces)
      self.loaded_model = joblib.load(self.loc + model)
      self.faceDetection = mp.solutions.face_detection.FaceDetection(model_selection=1)
      self.happy_numbers = utils.build_heatmap_table(self.loc, 'happy_dataset.csv')
      utils.count = 0
      self.not_happy_numbers = utils.build_heatmap_table(self.loc, 'not_happy_dataset.csv')
      

    def locateFacesInImage(self,img):
      imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
      results = self.faceDetection.process(imgRGB)
      bboxs = []
      if results.detections:
          for id,detection in enumerate(results.detections):
              bboxC = detection.location_data.relative_bounding_box
              ih, iw, ic = img.shape
              bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
                      int(bboxC.width * iw ), int(bboxC.height * ih)
              x, y, w, h = bbox
              crop_img = img[y:(y+h), x:(x+w)]
              bboxs.append([id, crop_img, bbox, detection.score])
      return bboxs


    def build_dataset_for_each_picture_with_face_detector(self, input_image, key, default_emution='happy'):
      data_points = utils.build_empty_column_table_data()
      detection_result = self.locateFacesInImage(input_image)
      try:   
          for detection in detection_result:
            try:
              results = self.mp_face_mesh.process(detection[1])
              if results.multi_face_landmarks:
                  for face_landmarks in results.multi_face_landmarks:
                      mesh_points = np.array([np.multiply([p.x, p.y, p.z], [1, 1, 1]).astype(float) for p in face_landmarks.landmark])
                      mesh_points = utils.transform_mesh(mesh_points)
                      for i,(x,y,z) in enumerate(mesh_points):
                          if (data_points.get(f'Point{i+1}_x') is not None):
                            if (self.happy_numbers.__contains__(i) and self.not_happy_numbers.__contains__(i)):
                                  data_points[f'Point{i+1}_x'].append(x)
                                  data_points[f'Point{i+1}_y'].append(y)
                            else:
                                  del data_points[f'Point{i+1}_x']
                                  del data_points[f'Point{i+1}_y']
                      data_points['Label'].append(default_emution)
                      df_2points = pd.DataFrame(data_points)
                      X_test = df_2points.drop('Label', axis=1)
                      y_test = df_2points['Label']
                      y_pred = self.loaded_model.predict(X_test)
                      score = accuracy_score(y_test, y_pred)
                      x, y, w, h = detection[2]
                      if (score == 1):
                        cv2.imwrite(self.loc + 'happy\\' + key + ".jpg", detection[1])
                        input_image = utils.addIcon(input_image,self.smile_icon, x-100,y-100)
                      # input_image = cvzone.overlayPNG(input_image,smile_icon,[x-100,y-100])
                      else:
                        cv2.imwrite(self.loc + 'not_happy\\' + key + ".jpg", detection[1])
                        input_image = utils.addIcon(input_image,self.not_smile_icon, x-100,y-100)
                        #input_image = cvzone.overlayPNG(input_image,not_smile_icon,[x-100,y-100])
                        
                      print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

              else:
                  print("Can't get mesh points for picture")
              data_points.clear()
              data_points = utils.build_empty_column_table_data()
            except Exception as e:
               print("Can't parse exception")
               self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3)
               print(e)
          cv2.imshow("Image Result", input_image)
      except Exception as e:
          print("Inside the exception....")
          print(e)               
   

    def image_capture(self, video, key,frame=5):
        video_capture = cv2.VideoCapture(video)
        frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        fpsPerSec = round(fps, 0)
        duration = round(( frames/fpsPerSec) , 0)
        fpsPerSec = fpsPerSec * int(frame) 
        count_pic = 1
        count_frames = 1
        while video_capture.isOpened():
          ret, img = video_capture.read()
          if not ret:
            break
          if count_frames%fpsPerSec==0:
            print("Movie time left:" + str(count_pic) + " out of:" + str(duration) + " seconds")
            self.build_dataset_for_each_picture_with_face_detector(img,key + "_" + str(count_frames))
            count_pic+=1
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          count_frames+=1
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--location", default=os.getcwd()+"\\",required=False,help="Location of the dataset")
    ap.add_argument("-m", "--model_file", default='golden_trained.pkl',required=False,help="trained model file name")
    ap.add_argument("-i", "--max_faces", default=3,type=int, required=False,help="Max number of faces to discovered")
    ap.add_argument("-v", "--video", required=True,help="Name of the video")
    ap.add_argument("-k", "--key", required=True,help="images key name saved on file system")
    ap.add_argument("-f", "--frame", default=5,required=False,type=int, help="Frame per second stop (default 5 sec)")
    args = ap.parse_args()
    movie_trainer = MovieTrainer(location=args.location, max_faces=args.max_faces,model=args.model_file)
    movie_trainer.image_capture(args.video, args.key, frame=args.frame)


