import urllib.request
import cv2
import numpy as np
import os

def store_raw_images():
    neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071'
    neg_image_url = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 72
    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_url.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i,"neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img,(100,100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num +=1
        except Exception as e:
            print(str(e))

def deleting_bad_images():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                
                try:
                    current_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_path)
            
                    if ugly.shape == question.shape and not (np.bitwise_xor(ugly,question).any()):
                        print('Deleting the image')
                        print(current_path)
                        os.remove(current_path)
                except Exception as e:
                    print(str(e))
                

def create_pos_n_negs():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            if file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f :
                    f.write(line)

                            
create_pos_n_negs()
#store_raw_images()
#deleting_bad_images()



