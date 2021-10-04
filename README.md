# Challenges in Deploying Deep Learning for Chest X-Ray Interpretation to Clinical Practice

## Project Summary
This Project is purposed to examine and leverage existing researches and models, to identify areas of concern or for improvement and to construct and evaluate one or more new model(s) by incorporating our research findings.  CheXpert dataset is utilized to train our CNN model with DenseNet121 as our baseline.  Among the 14 diagnosis labels specified by CheXpert, we focused on 5 disease diagnoses; Atelectasis, Cardiomegaly, Edema, Consolidation and Pleural Effusion.  AUC, False Postive Rate (FPR) and False Negative Rate (FNR) are metrics chosen to evaluate model performance. 

## Table of Contents
* [Dataset](#Dataset)
* [Architecture/Pipeline](#Architecture/Pipeline)
* [Models](#Models)
* [Results](#Results)
* [Setup](#Setup)
* [Code-Structure](#Code-Structure)

## Dataset
[CheXpert](https://arxiv.org/abs/1901.07031) dataset is comprised of 223,648 frontal and lateral CXR images for 64,740 patients, along with related text/csv files for labels and additional information such as gender, AP/PA, and filepath to patient CXR image.

***Note that direct link to dataset can not be provided due to competition regulations***

The dataset can be downloaded via registering and subscribing to [CheXpert website](https://stanfordmlgroup.github.io/competitions/chexpert/) (Stanford ML Group). There are two types of dataset available; low and high resolution versions. 


## Architecture/Pipeline
1. Scala and spark are used for partitioning the data based on our models
2. Python libraries used for image augmentations: 
    * cv2 (opencv-python)
    * albumentations
    * PIL
3. Pytorch for building and running model
    * Dense121 as baseline architecture
    * AWS for expensive computation
    

## Models
From the CheXpert data, we selected two datasets for model training and testing:
1. Full dataset, filtering out few records for data quality issues. 
2. 120K sampled patient studies; 70% for training and 30% for testing

Based on the two datasets, two models are trained and tested:
1. Replicated DenseNet121 from Pham et al.
2. Projection-based Ensemble

## Results
|                                                                                            | **AUC (All 14 labels)**    | **AUC (All 14 labels)**       |
|--------------------------------------------------------------------------------------------|----------------------------|-------------------------------|
| **population**<br>(Epoch=5)                                                                | **Replicated DenseNet121** | **Projection-based Ensemble** |
| Training: filtered CheXpert training data<br>Test: CheXpert validation set                 | 0.8708                     | 0.8588                        |
| Training: 70% of sampled 120k patient studies<br>Test: 30% of sampled 120k patient studies | 0.8468                     | 0.8531                        |


|                                 	|                 	|                  	| ***Diagnosis***   	|           	|                 	|             	|
|---------------------------------	|-----------------	|------------------	|-------------------	|-----------	|-----------------	|-------------	|
| **Model**                       	| **Atelectasis** 	| **Cardiomegaly** 	| **Consolidation** 	| **Edema** 	| **P. Effusion** 	| **Average** 	|
| Pham et al. Ensemble [14]       	| 0.909           	| 0.910            	| 0.958             	| 0.957     	| 0.964           	| 0.940       	|
| Pham et al. Single Model [14]   	| 0.825           	| 0.855            	| 0.937             	| 0.930     	| 0.923           	| 0.894       	|
| Replicated Pham DenseNet121     	| 0.821           	| 0.795            	| 0.898             	| 0.919     	| 0.932           	| 0.873       	|
| Projection-based Model Ensemble 	| 0.814           	| 0.788            	| 0.895             	| 0.901     	| 0.904           	| 0.860       	|
## Setup

### Environment Setup:

***Scala***

*Please note that all the scala processing were performed in Windows 10 setting.*

We selected ***IntelliJ*** as an integrated working space for scala and spark. Community version can be downloaded [here](https://www.jetbrains.com/idea/download/#section=mac). There are several pre-requisites need to be installed prior to building project in IntelliJ. 

* Java [JDK 1.8.0_271](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html) (We have used version 1.8.0_271).
* [Hadoop 3.2.1](https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.2.1/hadoop-3.2.1.tar.gz) (version 3.2.1 is used).
* Scala and SBT Executer plug-ins need to be installed in IntelliJ.

Once all the pre-requisites are installed, run 'Main.scala,' the scala object module in IntelliJ, which is located within 'Team68_Spark/src/main/scala/data_processor/' directory. 

Two folders will be created afterwards, where outputs will be generated into:
* Team68_Chexpert_Full
* Team68_Chexpert_Sample





***Python***

Jupyter Notebook was our primary platform where image augmentation and models training/testing were implemented. We recommend installing Anaconda to access Jupyter notebook to run our codes. Anaconda can be downloaded [here](https://docs.anaconda.com/anaconda/install/)
All the files and codes associated with training are included in 'Team68_Python' folder. 

It is crucial to ensure 'PyTorch' is installed in the system. Visit [Pytorch Installation Guide](https://pytorch.org/get-started/locally/) to install PyTorch. You can verify whether PyTorch is installed properly by:

```
$ python
>>> import torch
>>> print(torch.__version__)
```
**Basic Usage**

To execute the training, navigate to 'Team68_Python/traindn.py' and run the script by:
```
$ python traindn.py model_name predictions_name [args]
```
**Complete Guide**

```
usage: traindn.py [-h] [-c] [-a] [-g] [-s] [-e EPOCHS] [-w NUM_WORKERS]
                  [-b BATCH_SIZE]
                  msavename psavename

Train CheXpert DenseNet121 with best defaults

positional arguments:
  msavename             model savename postfix for saving improvement
                        checkpoints
  psavename             prediction savename postfix for saving post-train
                        predictions

optional arguments:
  -h, --help            show this help message and exit
  -c, --cuda            use cuda for training
  -a, --aws             training is on AWS
  -g, --grayscale       use grayscale images for training rather than
                        converting to color
  -s, --sample          use the scala processed sampled dataset for training
  -e EPOCHS, --epochs EPOCHS
                        num training epochs
  -w NUM_WORKERS, --workers NUM_WORKERS
                        num workers to use for dataloaders
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size for training and validation
```                    


## Code-Structure
```
├── README.md
├── Team68_Python
│   ├── chex
│   │   ├── __init__.py
│   │   ├── _experimental.py
│   │   ├── config.py
│   │   ├── dataaug.py
│   │   ├── dataload.py
│   │   ├── etl.py
│   │   ├── inference.py
│   │   ├── metrics.py
│   │   ├── modeling.py
│   │   ├── plotting.py
│   │   ├── saving.py
│   │   ├── training.py
│   │   └── utils.py
│   ├── data
│   │   └── templates
│   ├── experiment_notebooks
│   │   ├── BaseExperimentNB.ipynb
│   │   ├── experiments_batch_lr.ipynb
│   │   ├── experiments_data_fill.ipynb
│   │   ├── experiments_img_aug.ipynb
│   │   └── experiments_models.ipynb
│   ├── projection_ensembling.ipynb
│   ├── replication_modeling.ipynb
│   ├── supplemental_nbs
│   │   ├── Template_Matching_Image.ipynb
│   │   ├── modeling_part2.ipynb
│   │   ├── prelim_modeling.ipynb
│   │   ├── prelim_modeling_local.ipynb
│   │   └── thresholding.ipynb
│   └── traindn.py
└── Team68_Spark
    ├── build.sbt
    ├── data
    │   └── input
    ├── project
    │   ├── build.properties
    │   ├── project
    │   └── target
    ├── src
    │   └── main
    │       └── scala
    │           ├── data_processor
    │           │   └── Main.scala
    │           └── setup
    │               └── SparkSetup.scala
    └── target
        ├── scala-2.11
        ├── scala-2.12
        └── streams
```
