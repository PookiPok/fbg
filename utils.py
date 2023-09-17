import numpy as np
import pandas as pd
import cv2


count = 0
def transform_mesh(mesh_points):
    # Calculate the center of the mesh
    center = np.mean(mesh_points, axis=0)

    # Find the direction vector pointing from the center to a reference point (e.g., nose tip)
    reference_point = mesh_points[0]  # Change this to the appropriate reference point
    direction = reference_point - center

    # Calculate the rotation angle based on the direction vector
    rotation_angle = np.arctan2(direction[1], direction[0])

    # Create a rotation matrix using the rotation angle
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                [0, 0, 1]])

    # Apply the rotation to each vertex of the mesh
    rotated_mesh = np.dot(mesh_points - center, rotation_matrix) + center

    # Define the translation vector to move the mesh 50 cm away from the viewer
    # translation_vector = np.array([0, 0, -50])  # Adjust the values as needed
    translation_vector = np.array([0, 0, 0])  # Adjust the values as needed

    # Apply the translation to each vertex of the mesh
    transformed_mesh = rotated_mesh + translation_vector

    return transformed_mesh

def replace_column_names(column_name):
    global count
    count = count + 1
    return count

def build_std_tabel(dataset):
    df = pd.read_csv(dataset) 
    all_data = df.drop('Label', axis=1)

    all_data.columns = all_data.columns.map(replace_column_names)
    X_columns = [column for column in all_data.columns if column % 2 != 0]
    Y_columns = [column for column in all_data.columns if column % 2 == 0]

    new_df_X = all_data[X_columns]
    new_df_Y = all_data[Y_columns]

    point_x_1 = new_df_X[1]
    point_y_1 = new_df_Y[2]

    column_names = list(new_df_X)[1:]
    std_table = []
    std_table.append(0)
    for column in column_names:
        distance = np.sqrt(((point_x_1 - new_df_X[column])**2 + (point_y_1 - new_df_Y[column+1])**2))
        std_table.append(distance.std())
    
    return std_table

def build_empty_column_table_data():
  column_table = {}
  for i in range(0,468):
    column_table[f"Point{i+1}_x"]= []
    column_table[f"Point{i+1}_y"]= []
  column_table['Label'] = []
  return column_table

def build_heatmap_table(location, dataset, ratio=None):
    std_table = build_std_tabel(dataset)
    heatmap_numbers = []
    if (ratio == None):
        ratio = np.average(std_table)  
    for i,num in enumerate(std_table):
        if (num < ratio):
            heatmap_numbers.append(i)
    
    df = pd.DataFrame({"Number": [heatmap_numbers]})
    df.to_csv(location + 'num_' + dataset, index=False)
    return heatmap_numbers

def readCSV(dataset):
    try:
        df = pd.read_csv(dataset) 
        without_label = df.drop('Label', axis=1)
        only_label = df['Label']
        return without_label,only_label
    except Exception as e:
        print(e)

def addIcon(orig_img, icon, x, y):

  ht, wd = orig_img.shape[:2]
  ht2, wd2 = icon.shape[:2]

  # extract alpha channel as mask and base bgr images
  bgr = icon[:,:,0:3]
  mask = icon[:,:,3]

  bgr_new = orig_img.copy()
  if (y < 0):
     y = 0
  bgr_new[y:y+ht2, x:x+wd2] = bgr

  mask_new = np.zeros((ht,wd), dtype=np.uint8)
  mask_new[y:y+ht2, x:x+wd2] = mask
  mask_new = cv2.cvtColor(mask_new, cv2.COLOR_GRAY2BGR)
  result = np.where(mask_new==255, bgr_new, orig_img)
  return result
