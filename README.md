# RaSPDAM

## Introduction
**RaSPDAM** (**Ra**dio **S**ingle-**P**ulse **D**etection **A**lgorithm Based on Visual **M**orphological Features) is a novel machine learning algorithm specifically designed for the detection of Fast Radio Bursts (FRBs). Developed to address the computational and time challenges associated with traditional FRB search methods, RaSPDAM offers significant improvements in both efficiency and accuracy.

## Background
Fast Radio Bursts (FRBs) are brief, intense pulses of radio energy originating from distant galaxies. Their discovery and study are crucial for understanding the distribution and evolution of matter in the universe. However, the detection of FRBs is a challenging task due to the vast amounts of data produced by radio telescopes and the computational complexity of existing search methods. Traditional techniques often struggle with detecting weak signals and are time-consuming.

## Methodology
### Pre-processing
<div align="center">
  <img src="/pics/image_process.png" alt="Figure1" width="500">
  <p>Figure 1. Signal Preprocessing & Enhancement</p>
</div>

The preprocessing pipeline begins by converting the original signal sequence into standardized images without de-dispersion, where the sequence is divided into smaller slices using a 2-second sliding window with a 1-second overlap. These image slices are then resized to 512 × 512 pixels to standardize the model input. To enhance signal characteristics, particularly the typical curve of the FRB signal amidst noise, convolution is applied using multiple kernels generated from fixed curve slopes, transforming the image by processing each pixel and its local neighbors. After this, morphological dilation further enhances the signal. Finally, the enhanced images are combined with the original image to form an RGB image, which serves as the model input, as illustrated in Figure 1.

### Pulse searching
<div align="center">
  <img src="/pics/full_procedure.png" alt="Figure1" width="500">
  <p>Figure 2. Pulse Search Procedure</p>
</div>

RaSPDAM utilizes a U-Net architecture, a convolutional neural network (CNN) designed for image segmentation tasks. This U-Net model comprises an encoder that extracts slope-based features from the input image and a decoder that reconstructs the image using these features. Despite this process, some noise may still remain in the output, so we apply additional filtering steps to identify potential FRB signals. Using the "regionprops" function, we analyze connected regions in the segmented image and calculate each candidate's projections on the x- and y-axes. Candidates exceeding defined thresholds on both axes are flagged as potential FRB signals.

We have upgraded the model to use nnUNet, aligning it with NMS (Non-Maximum Suppression) and overlapping suppression techniques, in order to enhance performance.

For more details, please refer to the [paper](https://arxiv.org/abs/2411.02859).

## Dataset
The testing of RaSPDAM is facilitated by the **[FAST-FREX](https://www.scidb.cn/en/detail?dataSetId=3b3cf2f75a74419b89a56cc9626af2a0)** dataset, which is built upon observations obtained by the Five-hundred-meter Aperture Spherical radio Telescope (FAST). The dataset consists of:

- 600 positive samples: Observed FRB signals from three sources (FRB20121102, FRB20180301, and FRB20201124).
- 1000 negative samples: Noise and Radio Frequency Interference (RFI).

## Key Features
### Efficiency and Accuracy
- High Precision: RaSPDAM achieves a precision of 97.28%, significantly outperforming traditional methods like PRESTO and Heimdall.
- High Recall: With a recall rate of 83.50%, RaSPDAM effectively identifies a large proportion of true FRB signals.
- F1 Score: An F1 score of 0.9253 indicates a well-balanced trade-off between precision and recall.
### Versatility
- ToA and DM: While RaSPDAM currently provides Time of Arrival (ToA) as a result, future enhancements aim to include Dispersion Measure (DM) for more comprehensive signal verification.

## Performance Benchmarks
Comparison with Traditional Methods
|Software	|TN	|TP	|FN	|FP	|Recall	|Precision	|F1 Score|
|---------|---|---|---|---|-------|-----------|--------|
|PRESTO |3|472|0|26963700|0.7867|1.7505E-05|3.5009E-05|
|Heimdall|218|489|36|5854|0.8150|0.0771|0.1409|
|RaSPDAMv1(UNet)|989|466|128|6|0.7767|0.9873|0.8694|
|RaSPDAMv2(nnUNet)|994|501|67|14|0.8350|0.9728|0.9253|

## Discoveries
Since its deployment, RaSPDAM has been instrumental in identifying:
- 2 new FRBs: FRB20211103A and FRB20230104.
- 80 pulsars: Including 13 previously undiscovered pulsars, highlighting the algorithm's efficacy in uncovering new celestial objects.

___

# Usage:
## Training
### Generate Training Data
The following command can be used to generate training data in the running directory. Replace `{RaSPDAM_PATH}` with the actual path to the RaSPDAM directory. You can change the training data generation parameters in `{RaSPDAM_PATH}/utils/traning_data_gen.py`.
```shell
python {RaSPDAM_PATH}/utils/traning_data_gen.py
```
### Train the Model
Follow the instructions of nnUNet to train the model, refer to [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
- Create the `nnUNet_raw/{DatasetXXX_Name}` directory, place the generated training data and `{RaSPDAM_PATH}/models/dataset.json` in it, change the numTraining in json file. 
- Run `nnUNetv2_plan_and_preprocess` to generate the necessary preprocessing files.
- Run `nnUNetv2_train` to train the model. For FOLD in [0, 1, 2, 3, 4], run:
``` shell
nnUNetv2_train {DATASET_NAME_OR_ID} 2d {FOLD} --npz
```
- Copy the corresponding trained model `nnUNet_results/{DatasetXXX_Name}/nnUNetTrainer__nnUNetPlans__2d/` to the `{RaSPDAM_PATH}/models/` directory.


## Install Requirements
```shell
pip install -r requirements.txt
```

## Run the Script
```shell
python raspdam.py <fits_file> [-m <model_path>] [-o <output_path>] [-iou_theshold <value>] [-overlap_threshold <value>] [-box_fill_threshold <value>] [-box_projection_threshold <value>]
```
### Arguments：
- **fits_file**: The path to the FITS file to be processed (required).
- **-m, -model_path**: The path to the model file (optional, default: models).
- **-o, -output_path**: The path to save the output file (optional, default: current directory).
- **-iou_threshold**: Minimum IoU threshold for bounding box filtering (optional, default: 0.5).
- **-overlap_threshold**: Minimum overlap threshold for bounding box filtering (optional, default: 0.5).
- **-box_fill_threshold**: Minimum percentage threshold for the area filled within each bounding box (optional, default: 0.25).
- **-box_projection_threshold**: Minimum projection percentage threshold for bounding box filtering (optional, default: 0.15).