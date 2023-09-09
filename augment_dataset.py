import cv2
import imgaug as ia
import imageio
import shutil
import imgaug.augmenters as iaa
import os
ia.seed(1)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Augmentations
seq = iaa.Sequential([ 
    iaa.SaltAndPepper(0.01, per_channel=True),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), 
    iaa.Sharpen(alpha=(0.0, 0.4), lightness=(0.75, 1.0))

  
    ],random_order=True)

# Number of augmented image and mask to be generated

def augment_dataset(base_folder, output_img_folder, output_mask_folder, multiplication=3):

    input_folder = base_folder+"images/"
    input_mask_folder = base_folder+"masks/"

    for file in os.listdir(input_folder):
        filename,ext = os.path.splitext(file)



        for  c in range(multiplication):

            input_img_path = input_folder+ file
            input_mask_path = input_mask_folder+file


            output_img_path  = output_img_folder+filename+ "_"+str(c)+ext
            output_mask_path = output_mask_folder+filename+ "_"+str(c)+ext 


            if c == 0: #copy original images
                shutil.copy2(input_img_path, output_img_path)
                shutil.copy2(input_mask_path, output_mask_path)

            else:

                images_aug, masks_aug = seq(images=[cv2.imread(input_img_path)], segmentation_maps=[cv2.imread(input_mask_path)])
                cv2.imwrite(output_img_path, images_aug[0])
                cv2.imwrite(output_mask_path, masks_aug[0])


       
       


base_folder = "capillaries_dataset/" 

out_image_folder = base_folder+"augmented_images/"
out_mask_folder =base_folder+ "augmented_masks/"

if not os.path.exists(out_image_folder):
    os.makedirs(out_image_folder)

if not os.path.exists(out_mask_folder):
    os.makedirs(out_mask_folder)


augment_dataset(base_folder, out_image_folder, out_mask_folder)
