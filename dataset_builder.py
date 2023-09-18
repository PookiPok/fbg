import utils
import argparse
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


class DatasetBuilder:
  def __init__(self, image_static=True):
    self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=image_static)

  def build_heatmap_for_dataset(self, location, state='both'):
    total_pic = 0
    error_pic = 0   
    error_pic_path = []
    dataset_picture_directory_names = ['happy', 'not_happy']
    if (state == 'happy'):
        dataset_picture_directory_names = ['happy']
    elif (state == 'not_happy'):
        dataset_picture_directory_names = ['not_happy']

    data_table = utils.build_empty_column_table_data()
    dataset_file_name = state + '_dataset.csv'
    for dir_name in dataset_picture_directory_names:
        for subdir, dirs, files in os.walk(location + dir_name):
            for file in files:
                try:
                    print("Analazing mesh for picture:" + file)
                    total_pic+=1
                    file_path = os.path.join(subdir, file)
                    input_image = cv2.imread(file_path)  
                    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    results = self.mp_face_mesh.process(input_image_rgb)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mesh_points = np.array([np.multiply([p.x, p.y, p.z], [1, 1, 1]).astype(float) for p in face_landmarks.landmark])
                            mesh_points = utils.transform_mesh(mesh_points)
                            for i,(x,y,z) in enumerate(mesh_points):
                                if (data_table.get(f'Point{i+1}_x') is not None):
                                    data_table[f'Point{i+1}_x'].append(x)
                                    data_table[f'Point{i+1}_y'].append(y)
                            data_table['Label'].append(state)
                    else:
                        print("Can't retrive mesh points for the following picture:" + file)
                        error_pic+=1
                        error_pic_path.append(file_path)
                except Exception as e:
                    print(e)
                    error_pic+=1
                    error_pic_path.append(file_path)
    df_points = pd.DataFrame(data_table)
    df_points.to_csv(dataset_file_name, index=False)
    print("Total pic:" + str(total_pic))
    print("Success pic:" + str(total_pic - error_pic))
    print("Error %:" + str((error_pic/total_pic)*100))
    print("Error pic list:" + str(error_pic_path))


  def build_dataset_for_trainer(self, location, state='both', test_dir_name=None):
    dataset_picture_directory_names = ['happy', 'not_happy']
    if (state != 'both'):
        dataset_picture_directory_names = [test_dir_name]

    happy_numbers = utils.build_heatmap_table(location, 'happy_dataset.csv')
    utils.count = 0
    not_happy_numbers = utils.build_heatmap_table(location, 'not_happy_dataset.csv')
    data_table = utils.build_empty_column_table_data()
    dataset_file_name = state + '_dataset.csv'
    for dir_name in dataset_picture_directory_names:
        for subdir, dirs, files in os.walk(location + dir_name):
            for file in files:
                try:
                    print("Analazing mesh for picture:" + file)
                    file_path = os.path.join(subdir, file)
                    input_image = cv2.imread(file_path)  
                    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    results = self.mp_face_mesh.process(input_image_rgb)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mesh_points = np.array([np.multiply([p.x, p.y, p.z], [1, 1, 1]).astype(float) for p in face_landmarks.landmark])
                            mesh_points = utils.transform_mesh(mesh_points)
                            for i,(x,y,z) in enumerate(mesh_points):
                                if (data_table.get(f'Point{i+1}_x') is not None):
                                    if (happy_numbers.__contains__(i) and not_happy_numbers.__contains__(i)):
                                        data_table[f'Point{i+1}_x'].append(x)
                                        data_table[f'Point{i+1}_y'].append(y)
                                    else:
                                        del data_table[f'Point{i+1}_x']
                                        del data_table[f'Point{i+1}_y']
                            data_table['Label'].append(state)
                    else:
                        print("Can't retrive mesh points for the following picture:" + file)
                except Exception as e:
                    print(e)
    df_points = pd.DataFrame(data_table)
    df_points.to_csv(dataset_file_name, index=False)


if __name__ == "__main__":
    dataset_builder = DatasetBuilder()
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--location", default=os.getcwd() + '\\',required=False,help="Root picture directory location")
    ap.add_argument("-s", "--state", default='both', required=False,help="Which dataset to build, happy, not_happy and both (default)")
    ap.add_argument("-t", "--test_dir", required=False,help="Which test directory to use")

    args = ap.parse_args()
    if (args.state != 'both'):
        dataset_builder.build_heatmap_for_dataset(args.location, args.state)
    else:
        dataset_builder.build_dataset_for_trainer(args.location, args.state, args.test_dir)
