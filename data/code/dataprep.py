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
IMG_LIST   = os.path.join(ANNO_DIR,"img_list.txt")
BOX3D_LIST  = os.path.join(ANNO_DIR,"bbox3d.txt")
LABEL_FILE = os.path.join(ANNO_DIR,"labels.txt") # TODO

TCIA_DIR   = os.path.join(DATA_DIR,"tcia");

TEMP       = os.path.join(DATA_DIR,"temp")
NRRD_DIR   = os.path.join(DATA_DIR,"cropped_nrrds")

if not os.path.isdir(TEMP):
    os.mkdir(TEMP)

if not os.path.isdir(NRRD_DIR):
    os.mkdir(NRRD_DIR)

    
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

def show_slices(nrrd_file):
    img3d = read_nrrd(nrrd_file)
    
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

def crop3D(img_name,bbox):
    img_name = img_name.replace("_","-")
    print("Crop %s ..."%(img_name))

    x = bbox[0]
    y = bbox[1]
    z = bbox[2]

    lx = bbox[3]
    ly = bbox[4]
    lz = bbox[5]
    
    cimg = np.zeros((lx,ly,lz))
    full_size_nrrd = os.path.join(TEMP,img_name)+".nrrd"
    
    if not os.path.isfile(full_size_nrrd):
        print(f"{img_name} is not found")
        return

    img3d = read_nrrd(full_size_nrrd)
    #print(cimg.shape)
    #print((z,z+lz))
    for zi in range(lz):
        img = img3d[x:x+lx,y:y+ly,z+zi]
        img = np.flip(img,axis=0)
        img = np.flip(img,axis=1)

        cimg[:,:,zi] = img

    # now save it to nrrd
    nrrd_file = os.path.join(NRRD_DIR,img_name+"_Cropped_Volume.nrrd")
    nrrd.write(nrrd_file, cimg)
    
def crop():
    print("Start cropping images ...")
    f = open(BOX3D_LIST,"r")
    while True:
        line = f.readline().strip()
        if line is None or len(line)==0:
            break
        data = line.split(",")
        #print(data)
        img_name = data[0]
        bbox = [int(data[1]),int(data[2]),int(data[3]),int(data[4]),int(data[5]),int(data[6])]
        crop3D(img_name,bbox)

    print("Complete cropping images ...")

    
def dicom2nrrd(img_name,img_path):
    dicom_dir = os.path.join(TCIA_DIR,img_path)
    nrrd_file = os.path.join(TEMP,img_name+".nrrd")
    
    imreader = sitk.ImageSeriesReader()
    s_IDs    = imreader.GetGDCMSeriesIDs(dicom_dir)

    if not s_IDs:
        print(f"{dicom_dir} is not found!!!")
        return

    files = imreader.GetGDCMSeriesFileNames(dicom_dir, s_IDs[0])
    imreader.SetFileNames(files)
    image = imreader.Execute()

    sitk.WriteImage(image, nrrd_file)
    print(f"Saved NRRD: {nrrd_file}")

    
    
def extract():
    freader = open(IMG_LIST,'r')
    while True:
        line = freader.readline()
        if line is None or len(line)==0:
            break

        # convert dicom to rnn
        imname,impath = line.split(",")
        print(f"Processing {imname} ... ")
        #if not imname=="HN-CHUS-071":
        #    continue
        
        dicom2nrrd(imname,impath.strip())
        print(f"DICOM to .NRRD conversion completed!") 
        
        
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
    answer  = input("Do you want to delete the full-size .nrrd files (Y/N)?")
    if answer.lower()=="y" or answer.lower()=="yes":
        clear()


#run()

#img_name = "HN-CHUS-071" # Select an image name 
#show_slices(os.path.join(NRRD_DIR,img_name+"_Cropped_Volume.nrrd"))
