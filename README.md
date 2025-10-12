# CTData Preparation for Laryngeal CT Imaging

Despite the rich body of research in head and neck cancer, laryngeal cancer imaging lacks a dedicated public dataset for model development and comparison. We introduce the first curated laryngeal cancer dataset derived from six prominent Cancer Image Archive (TCIA) collections. To support efficient model training and reduce computational overhead, we extracted 1 mm isotropic volumes centered on the laryngeal region, extending from the epiglottis to the cricoid cartilage. Comprising 1,029 computed tomography (CT) images, our benchmark dataset is constructed through a reproducible and fully documented workflow. Though direct image sharing is restricted by TCIA licensing, we release detailed cropping parameters and an open-source image processing pipeline to regenerate the dataset from TCIA originals. This work aims to facilitate reproducible research in segmentation, classification, and prognostic modelling for laryngeal cancer by promoting collaboration and progress in this underrepresented domain.

### STEP 0: Download Datasets from https://www.cancerimagingarchive.net

We select images from six datasets in the Cancer Image Archive https://www.cancerimagingarchive.net. Downloading datasets from TCIA requires license to access Restriced Access data, which should be applied through proper channel. Users are encouraged to register and download 6 public datasets containing computed tomography (CT) scans of patients with confirmed head and neck cancer, including laryngeal cancer. The datasets include: RADCURE, Head-Neck-PET-CT, HEAD-NECK-RADIOMICS-HN1, HNSCC, Head-Neck-3DCT-RT, and QIN-HEADNECK.  All imaging data and associated metadata were downloaded using the National Biomedical Imaging Archive (NBIA) Data Retriever. Clinical annotations included age, sex, tumour subsite, T-stage, and treatment type (surgery, radiotherapy, or chemoradiotherapy), when available. The stage-wise breakdown of the laryngeal cancer cases across the different datasets is summarized in table below.

!<img width="782" height="333" alt="image" src="https://github.com/user-attachments/assets/11656225-b838-4a8b-af15-0fce2add419d" />
[image](imgs/dataset_table.png)

After downloading RADCURE, Head-Neck-PET-CT, HEAD-NECK-RADIOMICS-HN1, HNSCC, Head-Neck-3DCT-RT, and QIN-HEADNECK, users should put them under the "data/tcia" folder. If the "tcia" folder does not exist under the "data" folder, please create one.


NOTE: Make sure you place the downloaded TCIA data folder into the "data" directory and rename it as "tcia"

## Install dependencies

pip install SimpleITK pydicom pynrrd numpy 

## Run
python dataprep.py

### The result will be a set of cropped 3D images in .nrrd formats, stored in "cropped_nrrds" folder. We also provide the labels for classification to different stages in "data/annotations/LaryngealCT_metadata.xlsx". This would help to train a deep learning model.

# Run deep learning models
Management of laryngeal cancer relies greatly upon accurate staging. Advanced laryngeal cancer (stage T4) are most often managed by surgical removal of larynx and the other (non-T4) cases are given chemoradiotherapy. Hence, we attempted binary classification (T4 vs Non-T4) of laryngeal cancer in this benchmarking experiment. Preprocessing steps included Hounsfield Unit (HU) clipping in the range (-300,300), Z-score normalization and the cropped images were already resampled to 1mm voxel spacing. Six different data augmentations such as random affine transformations, elastic deformations, gamma correction, Gaussian noise and left-right flipping were done to improve the robustness of 434
the model. All models were trained using the Focal Loss function to emphasize the minority (T4) class, with α=[1.0, 436
4.0] and γ=2.0. Optimization was performed using Adam (learning rate = 1e-4), with early stopping (patience = 10 epochs) based on the best F1-score on the validation set five fold cross-validation technique was used to benchmark the performance of five 3D deep learning architectures—3D CNN, ResNet18, ResNet50, ResNet101, and DenseNet121on this binary classification of laryngeal cancer cases.

We provide 5 examples for 5 different deep learning models, which can be found in the "classification" folder.

3DCNN_benchmarking.ipynb

DenseNet121_benchmarking_MICCAI.ipynb

ResNet18_benchmarking.ipynb

ResNet50_benchmarking.ipynb

ResNet101_benchmarking.ipynb

We also provide our trained models [here](https://drive.google.com/drive/folders/12GX54a3H0-f7VdbeWA9dXziWfezF80Fd?usp=sharing).
