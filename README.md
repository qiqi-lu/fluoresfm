# Foundation model for fluorescence microscopic image restoration (FluoResFM)
**FluoResFM** is a foundation model for fluorescence microscopic image restoration, which can complete three tasks, including **denoising**, **deconvolution**, and **super-resolution**, in a unified model.
It is an all-in-one model that was trained on diverse fluorescence microscopic images in a supervised manner. The training data covering more than 20 types of biological structures, such as CCP, MT, ER, F-actin, lysosome, and so on.
It can be finetuned on new datasets to improve the performance.
The model can serve as a preprocessing tool for cellular and intracellular image analysis, such as cell and organelle segmentation.

### napari plugin
We have developed a user-frendly `napari` plugin for FluoResFM ([napari-fluoresfm](https://github.com/qiqi-lu/napari-fluoresfm)), which can accomplish the image patching, text embedding, model training, model fine-tuning, and model prediction in a interactive interface.

This repository contains the code for `data processing`, `model training and finetune`, `model evaluation`, and `results analysis`.

## DEPENDENCIES
The study was developed and tested on `WSL2` with `Ubuntu 24.04`
- Distributor ID: `Ubuntu`
- Description:    `Ubuntu 24.04.1 LTS`
- Release:        `24.04`
- Codename:       `noble`

We recommend using the Linux distribution to run the code. Because the `pytroch` and `triton` packages are more stable on Linux, and less GPU memory and faster training speed on Linux compared with Windows.

The codes were only tested on `python 3.11.9`. Besides, the following packages are required (mainly):
- `torch` : should be installed based on your CUDA version and system. $\Rightarrow$ [pytorch](https://pytorch.org/get-started/previous-versions/)
- `torchvision`
- `tensorboard`
- `triton-windows`: if you are using Windows, you need to install the `triton` package manually. $\Rightarrow$ [trinton-windows](https://github.com/lwylab/triton-windows)
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-image`
- `tqdm`
- `pydicom`
- `cellpose` : Cellpoase-SAM was used for cell/nuclei segmentation. $\Rightarrow$ [cellpose](https://github.com/MouseLand/cellpose)
- `nellie` : Nellie was used for intracellular organelle segmentation. $\Rightarrow$ [nellie](https://github.com/aelefebv/nellie)
- `flash-attn`
- `huggingface-hub` : huggingface was used for download `BiomedCLIP` model.
- `transformers`
- `open_clip_torch`
- `openpyxl`
- `tifffile`

The text encoder from [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) was employed as the text embedding model, the version used in this sduy can be download using the code in `download_embedder.py`.

## MODEL ARCHITECTURE
**FluoResFM** is based on a text-conditioned U-Net architecture, with a low-quality raw image and its correspongding text prior information (including task type, imaging object, and imaging condition) as inputs, a restored high-quality image as output.


![unet](/markdown/figures/unet.png)
**Detailed architecture of FluoResFM.** FluoResFM takes the low-quality image and its corresponding textual prior information as inputs and outputs the restored high-quality image. It employs a text-conditioned U-Net as backbone, which consists of an encoder, a decoder, and skip connections. The encoder begins with a convolution layer, and the decoder ends with a convolution layer to generate the restored images. Each scale of the encoder and decoder contains a residual block and a text-image fusion block. The text prompt is projected into a text embedding using a text embedder and then injected into the cross-attention layer within the text-image fusion block. Only blocks labeled with “fire” markers are fine-finetuned during the fine-tuning stage.

## DATA COLLECTION
All the data used for training and testing are publicly accessible.
The original datasets can be download from the following links:

#### Internal datasets for training and evaluation
| Dataset    | Task       | Image Object         | Microscopy            | Link                                                                 |
|------------|-----------|----------------------|----------------------|----------------------------------------------------------------------|
| BioSR      | DN, DCV, SR | CCP                  | WF, SIM              | [Link](https://figshare.com/articles/dataset/BioSR/13264793)       |
|            |           | ER                   | WF, SIM              | [Link](https://figshare.com/articles/dataset/BioSR/13264793)       |
|            |           | F-actin              | WF, SIM              | [Link](https://figshare.com/articles/dataset/BioSR/13264793)       |
|            |           | MT                   | WF, SIM              | [Link](https://figshare.com/articles/dataset/BioSR/13264793)       |
| BioSR+     | DN        | CCP                  | WF                   | [Link](https://zenodo.org/records/7115540)                         |
|            |           | ER                   | WF                   | [Link](https://zenodo.org/records/7115540)                         |
|            |           | F-actin              | WF                   | [Link](https://zenodo.org/records/7115540)                         |
|            |           | MT                   | WF                   | [Link](https://zenodo.org/records/7115540)                         |
|            |           | myosin-IIA           | WF                   | [Link](https://zenodo.org/records/7115540)                         |
| CARE       | DN        | nuclei (planaria)    | confocal             | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            |           | nuclei (Tribolium)   | multi-photon         | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            | SR        | histone (Drosophila) | light-sheet          | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            |           | nuclei (retina)      | confocal             | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            |           | nuclear-envelope     | confocal             | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            |           | nuclei/membrane      | two-photon           | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            | DCV       | granule              | WF                   | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            |           | tubulin              | WF                   | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
|            |           | tubulin              | confocal             | [Link](https://doi.org/10.17617/3.FDFZOF)                          |
| DeepBacs   | DN        | nucleoid             | confocal             | [Link](https://github.com/HenriquesLab/DeepBacs/wiki)              |
|            |           | MreB                 | confocal             | [Link](https://github.com/HenriquesLab/DeepBacs/wiki)              |
|            | DCV, SR   | membrane (E. coli)   | WF, SIM              | [Link](https://github.com/HenriquesLab/DeepBacs/wiki)              |
|            |           | membrane (S. aureus) | WF, SIM              | [Link](https://github.com/HenriquesLab/DeepBacs/wiki)              |
| W2S        | DN, DCV, SR| Mito                 | WF, SIM              | [Link](https://zenodo.org/records/3895807)                         |
|            |           | lysosome             | WF, SIM              | [Link](https://zenodo.org/records/3895807)                         |
|            |           | F-actin              | WF, SIM              | [Link](https://zenodo.org/records/3895807)                         |
| SR-CACO-2  | DN, SR    | histone              | confocal             | [Link](https://github.com/sbelharbi/sr-caco-2)                     |
|            |           | survivin             | confocal             | [Link](https://github.com/sbelharbi/sr-caco-2)                     |
|            |           | tubulin              | confocal             | [Link](https://github.com/sbelharbi/sr-caco-2)                     |
| FMD        | DN        | nuclei (BPAE)        | confocal             | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | F-actin (BPAE)       | confocal             | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | Mito (BPAE)          | confocal             | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | NCC (zebrafish)      | confocal             | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | chromosome (mice)    | confocal             | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | chromosome (mice)    | two-photon           | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | nuclei (BPAE)        | two-photon           | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | F-actin (BPAE)       | two-photon           | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | Mito (BPAE)          | two-photon           | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | nuclei (BPAE)        | WF                   | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | F-actin (BPAE)       | WF                   | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
|            |           | Mito (BPAE)          | WF                   | [Link](https://github.com/yinhaoz/denoising-fluorescence)          |
| VMSIM5     | DCV, SR   | Mito                 | WF, SIM              | [Link](https://zenodo.org/records/3295829)                         |
| RCAN-in    | DCV, SR   | MT                   | confocal, STED       | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)            |
|            |           | NPC                  | confocal, STED       | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)            |
|            | DN        | golgi                | SIM                  | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)            |


#### External datasets for evaluation
| Dataset   | Task       | Image Object | Microscopy | Link                                                               |
|-----------|-----------|--------------|------------|--------------------------------------------------------------------|
| SIM       | DCV       | F-actin      | WF, SIM    | [Link](http://www.cellimagelibrary.org/images/7052)              |
|           |           | MT           | WF, SIM    | [Link](http://www.cellimagelibrary.org/images/36797)             |
| VMSIM3    | DCV, SR   | Mito         | WF, SIM    | [Link](https://zenodo.org/records/3295829)                       |
| VMSIM488  | DCV       | microsphere  | WF, SIM    | [Link](https://zenodo.org/records/3295829)                       |
| VMSIM568  | DCV       | microsphere  | WF, SIM    | [Link](https://zenodo.org/records/3295829)                       |
| VMSIM647  | DCV       | microsphere  | WF, SIM    | [Link](https://zenodo.org/records/3295829)                       |
| RCAN-ex   | DN        | F-actin      | iSIM       | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)          |
|           |           | ER           | iSIM       | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)          |
|           |           | lysosome     | iSIM       | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)          |
|           |           | Mito         | iSIM       | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)          |
|           |           | MT           | iSIM       | [Link](https://zenodo.org/records/4624364#.YF4lBa9Kgal)          |
| BioTISR   | DN, DCV, SR| CCP          | WF, SIM    | [Link](https://zenodo.org/records/14760518)                      |
|           |           | F-actin      | WF, SIM    | [Link](https://zenodo.org/records/14760518)                      |
|           |           | lysosome     | WF, SIM    | [Link](https://zenodo.org/records/14760518)                      |
|           |           | MT           | WF, SIM    | [Link](https://zenodo.org/records/14760518)                      |
|           |           | Mito         | WF, SIM    | [Link](https://zenodo.org/records/14760518)                      |

#### Datasets used for segmentation evaluation
| Dataset       | Task       | Image Object          | Microscopy   | Link                                                              |
|--------------|-----------|-----------------------|-------------|-------------------------------------------------------------------|
| CARE         | DN        | nuclei (planaria)     | confocal    | [Link](https://doi.org/10.17617/3.FDFZOF)                        |
|              |           | nuclei (Tribolium)    | multi-photon| [Link](https://doi.org/10.17617/3.FDFZOF)                        |
|              | SR        | nuclei/membrane       | two-photon  | [Link](https://doi.org/10.17617/3.FDFZOF)                        |
| DeepBacs     | DN        | nucleoid              | confocal    | [Link](https://github.com/HenriquesLab/DeepBacs/wiki)            |
|              | DCV       | membrane (E. coli)    | WF, SIM     | [Link](https://github.com/HenriquesLab/DeepBacs/wiki)            |
|              |           | membrane (S. aureus)  | WF, SIM     | [Link](https://github.com/HenriquesLab/DeepBacs/wiki)            |
| SR-CACO-2    | DN, SR    | histone               | confocal    | [Link](https://github.com/sbelharbi/sr-caco-2)                   |
| FMD          | DN        | nuclei (BPAE)         | confocal    | [Link](https://github.com/yinhaoz/denoising-fluorescence)        |
|              |           | F-actin (BPAE)        | confocal    | [Link](https://github.com/yinhaoz/denoising-fluorescence)        |
|              |           | Mito (BPAE)           | confocal    | [Link](https://github.com/yinhaoz/denoising-fluorescence)        |
| BioTISR      | SR        | CCP                   | WF, SIM     | [Link](https://zenodo.org/records/14760518)                      |
|              |           | F-actin               | WF, SIM     | [Link](https://zenodo.org/records/14760518)                      |
|              |           | lysosome              | WF, SIM     | [Link](https://zenodo.org/records/14760518)                      |
| HL60         | DN, DCV   | nuclei                | confocal    | [Link](https://bbbc.broadinstitute.org/BBBC024)                  |
| Scaffold-A549| DN        | nuclei                | confocal    | [Link](https://github.com/Kaiseem/Scaffold-A549)                 |
| GranuSEG     | DN        | nuclei                | confocal    | [Link](https://cbia.fi.muni.cz/datasets/)                        |
| ColonTissue  | DCV       | nuclei                | confocal    | [Link](https://cbia.fi.muni.cz/datasets/)                        |

The processed datasets are available upon request from the corresponding author and can be provided via physical transfer if necessary.

## DATA PROCESSING
### DATA PREPARATION
#### images and patches
- `data_cleaning.py` : exclude the patches with only background noise.
- `count_patches.py` : count the number of patches in each dataset.
- `display_patch_dsitribution.py`: display the distribution of patches in each task and structure.
- `check_file_existence.py`: check if the patches exist in the dataset.
- `display_patches.py`: display the patches in the dataset to check visually.
- `display_images_test.py`: display the images in the internal and external test datasets.
- `test_utils_function.py`: test the utils functions used for data preprocessing and data fold and unfold.
- `test_model.py`: check the usability of the model and print the information of model.
#### text
- `text_generation.py` : generate the text for each datasets using the information in xlsx file.
- `download_embedder.py` : download the text embedder from huggingface.
- `text_embedding.py` : embed the text using the text embedder.
#### data information
- `excel_datasets.py` : curate the information of each dataset (e.g. dataset name, task, structure, number of samples, etc.).


## MODEL TRAINING AND FINETUNE
- `dataset_analysis.py` : consists of the dataset name dictionaries for internal and external datasets, finetune dataset, datasets shown in the radar plot, and daatsets for evaluating sementation performance.
- `train_it2i.py` : train or finetune the `FluoResFM` model on the internal/externla datasets (`i` refers to `image`,`t` refers to `text`).
- `train_i2i.py` : train or finetune the model without inputing the text.
- `test_it2i.py` : test the `FluoResFM` model on internal/external datasets. 
- `test_i2i.py` : test the model without inputing the text.
- `test_it2i_finetune.py` : test the finetuned model on sepcific dataset. Need specify the model name and checkpoint name in the `finetune_checkpoints.py` file.

## MODEL EVALUATION
- `result_evaluate.py` : evalute the results of each model on each dataset (calculate the metrics of each sample).
- `result_evaluate_finetune.py` : evalute the results of each model on each dataset in the finetune stage (calculate the metrics). Need specify the dataset name and model name in the `finetune_evaluation_methods.py` file. 
- `collect_results_all_samples.py` : collect the metric value of all samples into a excel file. Each task has a single excel file.
- `analysis_results_each_dataset.py` : analysis the results of each model on each dataset (calculate the mean and std of each metric)

## RESULTS ANALYSIS
#### image restoration
- `display_metrics_each_dataset.py` : display the metrics of each dataset using box plot.
- `display_each_sample.py` : display the restored image (and its error) of each sample in the internal and external datasets.
- `display_images.py` : display the images restored by different methods.
- `display_radar.py` : display the radar plot of internal and external datasets.
- `display_voilin.py` : display the voilin plot of internal and external datasets across different tasks.
- `display_image.py` : display the images restored by different methods in the internal and external datasets.

#### image segmentation
- `segment.py` : segment the images using the existed model. Cellpos-SAM to segment the cells and nuclei, and Nellie to segment the organelle.
- `evaluate_seg.py` : evaluate the segmentation results using the metrics.
- `collect_seg_metrics.py` : collect the segmentation metrics of each dataset. (calculate the mean and std of each metric)
- `display_mask_metrics.py` : display the segmentation metrics of each dataset using scatter plot with error bars.
- `display_mask_each.py` : display the segmentation results of each sample in the internal and external datasets.
- `display_mask.py` : display segmentation results of selected samples in the internal and external datasets.

#### finetune
- `display_finetune_metrics.py` : display the metrics of the finetuned models. Mutlple datasets, three metrics.
- `display_finetune_training_curve.py` : display the training curve of the finetuned models. Single dataset. To present the overfitting phenomenon.
- `display_finetune_num_sample.py` : display the effect of numbder of samples used for finetuning.

### Other functions
- `substract_background.py` : substract the background of the images using rolling-ball algorithm.
- `image_denoising.py` : denoise image using conventional denoising algorithms, such as non-local means.

## LICENSE
This project is licensed under the MIT License - see the LICENSE file for details.
