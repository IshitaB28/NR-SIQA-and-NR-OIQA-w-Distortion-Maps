# NR-SIQA-and-NR-OIQA-w-Distortion-Maps

This folder contains all the code required to run our experiments.

The required packages to be installed can be found in requirements.txt. 

We encourage the readers to download the original datasets from the respective websites as given in the papers where each dataset was proposed. The original works have been cited in our paper.

List of scripts:

1. "gen_data.py":  This is the function that is used to generate synthetically distorted images using pristine images from LIVE-pristine-dataset (http://live.ece.utexas.edu/research/quality/pristinedata.zip). Owing to the large size of the data, we have not been able to include the generated data in this folder. However, please find some sample images generated using the code in the "sample-generated-data" folder.

2. "unet_light_model.py": This script contains our proposed UNet model that is used to generate SSIM-like distortion map generation.

3. "train_unet_light.py": Run this script to train the UNet model on the synthetically generated distorted data. The saved weights for patch size 32 is given in this folder as "best_unet_light_32_patch_distortion_pristine_all.pth".

4. "test_unet_light.py": can be used to visually test the generated distortion maps by UNet model. Please note that the customized dataloaders in this script are solely for ease of generation of the images. They are different from those in "load_dataset.py" given below.

5. "load_dataset.py": This is the script that contains customized dataset loaders for each of the datasets. These will be used to train and test the quality assessment models.

P.S.: the 'mos.csv' files used for WIVC I and WIVC II datasets are created from their MOS.xlsx files. In order to get easy access to the individual quality scores of the stereopairs (as required by the Messai et. al. architecture), we have accumulated the scores in one .csv file each for WIVC I and WIVC II datasets. These csv files are shared in this folder as "WIVC_I_mos.csv" and WIVC_II_mos.csv".

6. "stereo_models.py": contains all the models that can be used for the quality assessment of stereoscopic images. In our study, they can be used for LIVE I, LIVE II, WIVC I, WIVC II, IRCCyn, and LIVE3DVR (stereoscopic omnidirectional) datasets.

7. "omni2d_models.py": contains all the models that can be used for the quality assessment of 2D images or 2D omnidirectional images. In our study, they can be used for CVIQ and IQA-ODI datasets.

8. "train_stereo_qa_models.py": used to train the models from "stereo_models.py" on stereoscopic image quality assessment databases.

9. "test_stereo_qa_models.py": tests the models from "stereo_models.py" on stereoscopic image quality assessment databases. This script can also be used for cross dataset results on SIQA datasets.

10. "train_omni2d_qa_models.py": used to train the models from "omni2d_models.py" on omnidirectional image quality assessment databases.

11. "test_omni2d_qa_models.py": tests the models from "omni2d_models.py" on omnidirectional image quality assessment databases. This script can also be used for cross dataset results on OIQA datasets.

The "patch_size" parameters in the scripts can be varied to obtain the results of the impact of patch size on the QNet. The alternate in configurations for the model are mentioned in comments for patch size 16, 32, 64, and 128

12. "double_cross_data.py": can be used to run cross dataset studies when the model is trained on two databases instead of one. The dataset classes have been modified in the script for training only such that the quality scores is brought within a common range.

Procedure to change the models based on requirements such has "images and distortion maps" or "images only" are mentioned in the script.

13. "test_gradcam.py": Run this script to visualize GRADCAM outputs. Examples for IRCCyn and LIVE I dataset are shown.

In order to check the inference times, the following scripts can be run:
14. "stereo_inference_time.py", for stereoscopic image quality assessment (SIQA) using both distortion maps and images,
15. "stereo_inference_time_image_only.py", for SIQA using just images
16. "omni2d_inference_time.py", for omnidirectional image quality assessment

The SIQA with and without distortion maps are put in separate scripts to avoid the computation of distortion maps in the latter case. Similarly for omni, the required code can be uncommented.

One sample stereoscopic pair and one omnidirectional image are shared for inference.

========================================================================================================================================================

The licenses of all the used packages are mentioned in the file "used_pkgs_license_list.txt". Further, the github links to the two SOTA models that we have experimented with are given here as well as in the code script as comments. They have also been cited in the paper.

1. O. Messai and A. Chetouani, “End-to-end deep multi-score model for No-reference stereoscopic image quality assessment,” in \emph{Proc. 2022 IEEE International Conference on Image Processing}, Oct. 2022, pp. 2721–2725.

https://github.com/o-messai/Multi-score-SIQA

MIT License

Copyright (c) 2022 Oussama Messai

2. W. Zhou, Z. Chen, and W. Li, “Dual-Stream Interactive Networks for No-Reference Stereoscopic Image Quality Assessment,” \emph{IEEE Transactions on Image Processing}, vol. 28, no. 8, pp. 3946–3958, Aug. 2019.

https://github.com/weizhou-geek/Stereoscopic-Image-Quality-Assessment-Network

Copyright (c) 2018 University of Science and Technology of China All rights reserved.
