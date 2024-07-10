import tensorflow as tf
import numpy as np
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from glob import glob



def escalar(data, escala, size = 128, batch = 256):

    Xtest_big = np.zeros((data.shape[0],size,size,1))
    a = (size-data.shape[1])//2
    Xtest_big[:,a:a+data.shape[1],a:a+data.shape[1]] = data[:,:,:]

    Xtest_escala = tf.image.resize(data, size=(int(data.shape[1]*escala), int(data.shape[2]*escala))).numpy()
    new_shape = Xtest_escala.shape[1:-1]
    Xtest_escala_big = np.zeros_like(Xtest_big)
    Xtest_escala_big[:,size//2-new_shape[0]//2:size//2+int(np.round(new_shape[0]/2)),size//2-int(np.round(new_shape[1]/2)):size//2+new_shape[1]//2,:] = Xtest_escala
    return tf.data.Dataset.from_tensor_slices((Xtest_big, Xtest_escala_big)).batch(batch)


def scale_mosaic(data, escala, size = (32,32), batch = 100):
    Xtest_escala = tf.image.resize(data, size=(int(data.shape[1]*escala), int(data.shape[2]*escala))).numpy()
    paddings = tf.constant([[0,0], [Xtest_escala.shape[1], Xtest_escala.shape[1],], [Xtest_escala.shape[2], Xtest_escala.shape[2]], [0, 0]])
    mosaico = tf.pad(Xtest_escala, paddings, mode = "SYMMETRIC")
    while mosaico.shape[1] < min(size):
        paddings = tf.constant([[0,0], [mosaico.shape[1], mosaico.shape[1],], [mosaico.shape[2], mosaico.shape[2]], [0, 0]])
        mosaico = tf.pad(mosaico, paddings, mode = "SYMMETRIC")

    b, h, w, c = mosaico.shape
    final_imagesize = size

    ini_crop1 = (h//2)-(final_imagesize[0]//2)
    ini_crop2 = (w//2)-(final_imagesize[1]//2)
    Xtrain_big = mosaico[:,ini_crop1:ini_crop1+final_imagesize[0],ini_crop2:ini_crop2+final_imagesize[1],:]

    # return tf.data.Dataset.from_tensor_slices((tf.cast(data, tf.float32), tf.cast(Xtrain_big, tf.float32))).batch(batch)
    # return tf.data.Dataset.from_tensor_slices((data, Xtrain_big)).batch(batch)
    return Xtrain_big


def create_mosaic(data):
    paddings = tf.constant([[0,0], [data.shape[1], data.shape[1]], [data.shape[2], data.shape[2]], [0, 0]])
    mosaico = tf.pad(data, paddings, mode = "SYMMETRIC")

    return mosaico


def rmse(img, # (B,H,W,C)
         img_desp, # (B,H,W,C)
         ): # (B,)
    return (tf.reduce_mean((img - img_desp)**2, axis = (1,2,3)))**(1/2)

def rotation(X, rotacion):
  image_center = tuple(np.array(X.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, rotacion, 1.0)
  result = cv2.warpAffine(X, rot_mat, X.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotation2(X, crop, rotacion):
  h, w, c = X.shape
  final_imagesize = crop
  ini_crop = ((h//2)-(final_imagesize[0]//2),(w//2)-(final_imagesize[1]//2))

  height, width = X.shape[0:2]
  image_center = (width//2, height//2)
  rot_mat = cv2.getRotationMatrix2D(image_center, rotacion, 1.0)
  result = cv2.warpAffine(X.numpy(), rot_mat, dsize=(width, height), flags=cv2.INTER_LINEAR)
  result = result[ini_crop[0]:ini_crop[0]+final_imagesize[0],ini_crop[1]:ini_crop[1]+final_imagesize[1],:]
  return result

def agrandar(X):
  X_big = np.zeros((X.shape[0],56,56,1))
  X_big[:,14:14+28,14:14+28] = X[:,:,:]
  return X_big

def traslation(X, desp_h, desp_v):
  X_big = np.zeros((X.shape[0],56,56,1))
  X_big[:,14+desp_v:14+28+desp_v,14+desp_h:14+28+desp_h] = X[:,:,:]
  return X_big

def traslation2(X, crop, desp_h, desp_v):
  b, h, w, c = X.shape
  final_imagesize = crop
  ini_crop = ((h//2)-(final_imagesize[0]//2),(w//2)-(final_imagesize[1]//2))

  X_big = X[:,ini_crop[0]+desp_h:ini_crop[0]+final_imagesize[0]+desp_h,ini_crop[1]+desp_v:ini_crop[1]+final_imagesize[1]+desp_v,:]
  return X_big
