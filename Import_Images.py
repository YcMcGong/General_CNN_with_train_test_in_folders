"""
Copyright (c) 2017 Yicong Gong <gongyc2@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Python script to read a number of images from each folder

"""
import matplotlib.image as mpimg 
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Import_Data:

    def __init__(self, path, sample_size,subject_list):
        self.sample_size = sample_size
        self.validate_size = 0.1
        self.train_cursor = 0
        self.test_cursor = 0
        self.image_cursor = 0
        self.img_train, self.img_test, self.label_train, self.label_test, self.images, self.labels = self.Import_Images(path,subject_list)
        self.num_examples = int(len(self.img_train))
        self.num_validate_examples = int(len(self.img_test))
        self.num_images = int(len(self.images))
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

    # Old shuffule
    # def Shuffule(self):
    #     self.img_train, self.img_test, self.label_train, self.label_test = train_test_split(self.images, self.labels, test_size=self.validate_size)
    #     pass

    # Shuffule to enable stochastic gradient descent
    def Shuffule(self):
        self.img_train, self.label_train = shuffle(self.img_train, self.label_train)
        pass

    def Import_Images(self, path, subject_list):
        subject_number = 0
        images = []
        labels = []
        for subject in subject_list:
            imgs, labs = self.load_images_from_folder(path+subject+"/", self.sample_size, subject_number)
            subject_number = subject_number + 1
            images.extend(imgs)
            labels.extend(labs)
        img_train, img_test, label_train, label_test = train_test_split(images, labels, test_size=self.validate_size, random_state = 22) #42
        return img_train, img_test, label_train, label_test, images, labels

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

    def next_img_batch(self, batch_size):
        temp_end_cursor = self.image_cursor + batch_size
        batch_img_out = self.images[self.image_cursor:temp_end_cursor]
        batch_label_out = self.labels[self.image_cursor:temp_end_cursor]
        self.image_cursor = temp_end_cursor
        return batch_img_out, batch_label_out
        

# img, lab = Import_Data("./Training_Data/")
# print (len(img))
# print(lab)
