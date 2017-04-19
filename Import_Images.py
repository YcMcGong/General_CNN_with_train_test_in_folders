"""
Python script to read a number of images from each folder

"""
import matplotlib.image as mpimg 
import os
import numpy as np
from sklearn.model_selection import train_test_split

class Import_Data:

    def __init__(self, path, sample_size,subject_list):
        self.sample_size = sample_size
        self.validate_size = 0.1
        self.train_cursor = 0
        self.test_cursor = 0
        self.img_train, self.img_test, self.label_train, self.label_test = self.Import_Images(path,subject_list)
        self.num_examples = int(len(self.img_train))
        self.num_validate_examples = int(len(self.img_test))
        print("Data Reading Finished \n")
        print("Training_Size: ",len(self.img_train),"Validate_Size: ",len(self.img_test))

    def load_images_from_folder(self, folder, sample_size, subject_number):
        images = []
        # for filename in os.listdir(folder):
        dir_list = os.listdir(folder)
        sample_number = 0
        file_number = 0
        while sample_number < sample_size:
            filename = dir_list[file_number]
            img = mpimg.imread(os.path.join(folder,filename))
            #In case an image is not in the required size
            if (np.shape(img) == (224,224,3)):
                images.append(img)
                sample_number = sample_number +1
            file_number = file_number + 1
        labels = (np.ones(sample_size, dtype=int)*subject_number)
        return images, labels

    def Import_Images(self, path, subject_list):
        subject_number = 0
        images = []
        labels = []
        for subject in subject_list:
            imgs, labs = self.load_images_from_folder(path+subject+"/", self.sample_size, subject_number)
            subject_number = subject_number + 1
            images.extend(imgs)
            labels.extend(labs)
        img_train, img_test, label_train, label_test = train_test_split(images, labels, test_size=self.validate_size)
        return img_train, img_test, label_train, label_test

    def next_train_batch(self, batch_size):
        if(self.train_cursor>=self.num_examples): self.train_cursor = 0
        temp_end_cursor = self.train_cursor + batch_size
        batch_img_out = self.img_train[self.train_cursor:temp_end_cursor]
        batch_label_out = self.label_train[self.train_cursor:temp_end_cursor]
        self.train_cursor = temp_end_cursor
        return batch_img_out, batch_label_out

    def next_validate_batch(self, batch_size):
        if(self.test_cursor>=self.num_validate_examples): self.test_cursor = 0
        temp_end_cursor = self.test_cursor + batch_size
        batch_img_out = self.img_test[self.test_cursor:temp_end_cursor]
        batch_label_out = self.label_test[self.test_cursor:temp_end_cursor]
        self.test_cursor = temp_end_cursor
        return batch_img_out, batch_label_out

# img, lab = Import_Data("./Training_Data/")
# print (len(img))
# print(lab)