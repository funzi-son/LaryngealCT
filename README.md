# CT Data Preparation for Laryngeal CT Imaging

Despite the rich body of research in head and neck cancer, laryngeal cancer imaging lacks a dedicated public dataset for model development and comparison. We introduce the first curated laryngeal cancer dataset derived from six prominent Cancer Image Archive (TCIA) collections. To support efficient model training and reduce computational overhead, we extracted 1 mm isotropic volumes centered on the laryngeal region, extending from the epiglottis to the cricoid cartilage. Comprising 1,029 computed tomography (CT) images, our benchmark dataset is constructed through a reproducible and fully documented workflow. Though direct image sharing is restricted by TCIA licensing, we release detailed cropping parameters and an open-source image processing pipeline to regenerate the dataset from TCIA originals. This work aims to facilitate reproducible research in segmentation, classification, and prognostic modelling for laryngeal cancer by promoting collaboration and progress in this underrepresented domain.

### STEP 0: Download Datasets from https://www.cancerimagingarchive.net

We select images from six datasets in the Cancer Image Archive https://www.cancerimagingarchive.net. Downloading datasets from TCIA requires license to access Restriced Access data, which should be applied through proper channel. Users are encouraged to register and download 6 public datasets containing computed tomography (CT) scans of patients with confirmed head and neck cancer, including laryngeal cancer. The datasets include: RADCURE, Head-Neck-PET-CT, HEAD-NECK-RADIOMICS-HN1, HNSCC, Head-Neck-3DCT-RT, and QIN-HEADNECK.  All imaging data and associated metadata were downloaded using the National Biomedical Imaging Archive (NBIA) Data Retriever. Clinical annotations included age, sex, tumour subsite, T-stage, and treatment type (surgery, radiotherapy, or chemoradiotherapy), when available. The stage-wise breakdown of the laryngeal cancer cases across the different datasets is summarized in table below.

<img width="782" height="333" alt="image" src="https://github.com/user-attachments/assets/11656225-b838-4a8b-af15-0fce2add419d" />

After downloading RADCURE, Head-Neck-PET-CT, HEAD-NECK-RADIOMICS-HN1, HNSCC, Head-Neck-3DCT-RT, and QIN-HEADNECK, users should save them under the "data/tcia" folder. If the "tcia" folder does not exist under the "data" folder, please create one.


NOTE: Make sure you place the downloaded TCIA data folder into the "data" directory and rename it as "tcia".

## Install dependencies

pip install SimpleITK pydicom pynrrd numpy 

## Run
python dataprep.py

### The result will be a set of cropped 3D images in .nrrd formats, stored in "cropped_nrrds" folder. We also provide the labels for classification to different stages in "data/annotations/LaryngealCT_metadata.xlsx". This would help to train a deep learning model.

# Run deep learning models
Management of laryngeal cancer relies greatly upon accurate staging. From the different T4 class labels available for the LaryngealCT dataset, we conducted two benchmarking experiments: 
(1) Early (Tis-T1-T2) vs Advanced (T3-T4), and (2) Non-T4 vs T4 laryngeal cancer.
Rationale behind classification tasks: 
(1)Early vs Advanced stages were classified to minimize the class imbalance in the available dataset (638 (62%) Early and 391 (38%) Advanced). This broader classification helps in risk-stratification.These experiments also demonstrate the benchmarking performance of DL models on LaryngealCT dataset in the absence of severe class imbalance (as in T4 benchmarking).
(2)NonT4 vs T4 classification is more clinically relevant and has direct impact on treatment decisions and patient outcomes. T4 stage laryngeal cancer is most often managed by surgical removal of larynx and the other (Non-T4) cases are given chemoradiotherapy. There was a severe class imbalance in this classification task as we had 945 (92%) Non-T4 cases and only 84 (8%) T4 cases. The benchmarking performance of DL models for this task showed low sensitivity for T4 class, attributable to very less number of the minority class samples. This experiment underpins the need for data augmentation for T4 class for future works.

Preprocessing steps included Hounsfield Unit (HU) clipping in the range (-300,300), Z-score normalization and the cropped images were already resampled to 1mm voxel spacing. The dataset (n=1029) was divided into 823 train and 206 test images using stratified sampling, to ensure the equal distribution of all T stages and cases from all six component datasets across the splits. Six different data augmentations such as random affine transformations, elastic deformations, gamma correction, Gaussian noise and left-right flipping were done to improve the robustness of the model. All models were trained from scratch using randomly initialized weights, using the Focal Loss function to emphasize the minority (T4) class with α=8.0 and γ=2.0. Optimization was performed using AdamW (learning rate = 1e-4), with early stopping (patience = 20 epochs) based on the best macroF1-score on the validation set. Five fold cross-validation technique was used in training to benchmark the performance of five 3D deep learning architectures— a custom 5-layered 3D CNN, ResNet18, ResNet50, ResNet101, and DenseNet121.To explore the performance of transfer learning, we also included a ResNet50 model trained on MedicalNet pretrained weights. The best model was tested on an independent test set (n=206).

We provide examples for 6 different deep learning models for each of the benchmarking tasks, which can be found in the "classification" folder. This folder contains the codes, cross validation, test and per-class metrics, model checkpoints for each model, along with ROC curves, Precision-Recall curves and confusion matrices.

Explainable AI is gaining popularity in medical imaging, and we also attempted GradCAM++ visualizations for explainability of the best performing model for our T4 classification task. Step-by-step codes, sample segmentation labels and the results are provided in the "GradCAM" sub-folder of "classification" folder.

