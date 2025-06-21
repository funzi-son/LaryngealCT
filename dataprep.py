import numpy as np
import nrrd
import matplotlib.pyplot as plt
import pydicom
import pydicom.data
import sys
import math
import glob
import os
import SimpleITK as sitk

DATA_DIR   = "./data"
ANNO_DIR   = os.path.join(DATA_DIR,"annotations")
IMG_LIST   = os.path.join(ANN_DIR,"img_list.txt")
BOX3D_DIR  = os.path.join(ANNO_DIR,"box3d")
LABEL_FILE = os.path.join(ANNO_DIR,"labels.txt") # TODO

TCIA_DIR   = os.path.join(DATA_DIR,"tcia");

TEMP       = os.path.join(DATA_DIR,"temp")
NRRD_DIR   = os.path.join(DATA_DIR,"cropped_nrrds")

def imsim(img1,img2,method="mse"):
    if method=="mse":
        img1 = (img1-np.min(img1))/(np.max(img1)-np.min(img1))
        img2 = (img2-np.min(img2))/(np.max(img2)-np.min(img2))

        return ((img1 - img2) ** 2).mean() ** 0.5
    
def show_slice(img3d,inds=None):
   
    if inds is None:
        inds = [img3d.shape[0]//2,img3d.shape[1]//2,img3d.shape[2]//2]
        
    a1 = plt.subplot(1, 3, 1)
    img = img3d[inds[0], :, :]
    plt.imshow(img,cmap='gray',aspect="auto")
                
    a3 = plt.subplot(1, 3, 2)
    img = img3d[:, inds[1], :]
    plt.imshow(img,cmap='gray',aspect="auto")

    a2 = plt.subplot(1, 3, 3)
    img = img3d[:, :, inds[2]]
    plt.imshow(img,cmap='gray',aspect="auto")
             
    plt.show()
        
def read_nrrd(nrrd_file):
    cropped3D, header = nrrd.read(nrrd_file,index_order="F")

    return cropped3D

def read_dicom(dcm_folder):
    # load the DICOM files
    files = []
    print(f"glob: " + dcm_folder)
    for fname in glob.glob(dcm_folder+"/*.dcm", recursive=False):
        #print(f"loading: {fname}")
        files.append(pydicom.dcmread(fname))

    print(f"file count: {len(files)}")

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print(f"skipped, no SliceLocation: {skipcount}")

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    #print(len(slices))

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]


    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d


    return img3d

def crop3D(img_name):
    pos,shape = read_3dbox(os.path.join(BOX3D_DIR,img_name))
    
    x = pos[0]
    y = pos[1]
    z = pos[2]

    cimg = np.zeros(shape)
    for zi in range(z,z+shape[2]):
        img = img3d[x:xi+shape[0],y:y+shape[1],zi]
        img = np.flip(img,axis=0)
        img = np.flip(img,axis=1)

        cimg[:,:,i] = img

    # now save it to nrrd
    # TODO, save the image to nrdd
    nrrd_file = os.path.join(NRRD_DIR,img_name+"_Cropped_Volume.nrrd")
    
def crop():
    box_names = glob.glob(BOX3D_DIR)

    for bname in box_names:
        if bname is in [".", ".."]:
            continue

        img_name = "" # TODO: GET THE NAME OF THE IMAGE
        crop3D(img_name)

    
def dicom2rnn(img_name):
    dicom_dir = os.path.join(TCIA_DIR,img_name)
    nrrd_file = os.path.join(TEMP,img_name+".nrrd")

    imreader = sitk.ImageSeriesReader()
    s_IDs    = imreader.GetGDCMSeriesIDs(dicom_dir)

    if not s_IDs:
        print(f"{dicom_dir} is not found!!!")
        return

    files = reader.GetGDCMSeriesFileNames(dicom_dir, s_IDs[0])
    imreader.SetFileNames(files)
    image = imreader.Execute()

    sitk.WriteImage(image, nrrd_file)
    print(f"Saved NRRD: {nrrd_file}")

    
    
def extract():
    freader = fopen(IMG_LIST,'r')
    while True:
        img_name = freader.readline()
        if img_name is None or len(img_name)==0:
            break

        # convert dicom to rnn
        dcom2rnn(img_name)
        
        
def clear():
    """
     Clear the temporary nrrd files
    """
    
    nrrd_files = glob.glob(TEMP+"/*.nrrd")
    
    for f in nrrd_files:
        os.path.delete(f) # DELETE TEMP
    
def run():
    # Select the image from tcia dataset and convert them to nrrd
    extract()
    
    # Crop the nrrd
    crop()        

    # Clear temporary data
    clear()


run()
