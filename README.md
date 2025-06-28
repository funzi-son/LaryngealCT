# CTData Preparation for Laryngeal CT Imaging

TODO: Write something to introduce the project

### STEP 0: Download Datasets from https://www.cancerimagingarchive.net

We select images from six datasets in the Cancer Image Archive https://www.cancerimagingarchive.net. Users are encouraged to register and download 6 public datasets containing computed tomography (CT) scans of patients with confirmed head and neck cancer, including laryngeal cancer. The datasets include: RADCURE, Head-Neck-PET-CT, HEAD-NECK-RADIOMICS-HN1, HNSCC, Head-Neck-3DCT-RT, and QIN-HEADNECK.  All imaging data and associated metadata were downloaded using the National Biomedical Imaging Archive (NBIA) Data Retriever. Clinical annotations included age, sex, tumour subsite, T-stage, and treatment type (surgery, radiotherapy, or chemoradiotherapy), when available. The stage-wise breakdown of the laryngeal cancer cases across the different datasets is summarized in table below.

![image](imgs/dataset_table.png)

After downloading RADCURE, Head-Neck-PET-CT, HEAD-NECK-RADIOMICS-HN1, HNSCC, Head-Neck-3DCT-RT, and QIN-HEADNECK, users should put them under the "data/tcia" folder. If the "tcia" folder does not exist under the "data" folder, please create one.


NOTE: Make sure you place the downloaded TCIA data folder into the "data" directory and rename it as "tcia"

## Install dependencies

pip install SimpleITK pydicom pynrrd numpy 

## Run
python dataprep.py
