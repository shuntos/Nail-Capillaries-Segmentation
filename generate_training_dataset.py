import cv2
import numpy as np 
import os
import json

input_folder = "training_images/"

base_folder = "capillaries_dataset/"

img_folder = base_folder+"images/"
mask_folder = base_folder+"masks/"

if not os.path.exists(img_folder):
	os.makedirs(img_folder)

if not os.path.exists(mask_folder):
	os.makedirs(mask_folder)

for file in os.listdir(input_folder):
    input_file = input_folder+file

    filename,ext = os.path.splitext(file)

    if ext == ".json":

        img_file = input_folder+filename+".jpg"
        img = cv2.imread(img_file)

        print(input_file)
        with open(input_file) as j_file:
            json_data = json.load(j_file)

            mask = np.zeros([img.shape[0],img.shape[1],3], dtype=np.uint8)

            for i in range(len(json_data["shapes"])):
                label = json_data["shapes"][i]["label"]
                print(label)
                points = json_data["shapes"][i]["points"]
                pts = np.array(points,dtype = "int")

                cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            out_file_mask = mask_folder+filename+".jpg"
            cv2.imwrite(out_file_mask,mask)
            out_file_jpg = img_folder+filename+".jpg"
            cv2.imwrite(out_file_jpg,img)
