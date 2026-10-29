# Preparing a ConvNeXt Model for Neuroimaging, AD vs NC Classification

This project uses a layer-by-layer PyTorch implementation of ConvNeXt-Tiny to classify ADNI MRI slices as either Alzheimer's disease (AD) or normal cognition (NC).
The model uses a central slice and its left and right adjacent slices of an MRI scan as a 2.5D input into the model and follows the official TorchVision model architecture without using a prebuilt model or using pretrained weights. The code for the ConvNeXt-Tiny model layers was taken and adapted from the pytorch/vision repository at https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py. OpenAI's GPT-5.6 Sol (`gpt-5.6-sol`) was used as a prompting and coding assistant during development to help draft prompts, implement and debug code, and refine the project documentation.

The implementation foreground-centres and resizes each MRI slice, applies foreground only normalisation to each input channel separately and image augmentation that is shared between input channels, and uses patient separated training and validation data splits from the supplied training set with diagnosis and patient balanced sampling utilising the metadata. ConvNeXt-Tiny is trained from scratch in batches of eight using AdamW, label smoothing, stochastic depth, mixed precision, gradient clipping and MixUp, while exponential moving-average weights are used for validation, checkpoint selection and final inference. The learning rate uses linear warm-up followed by reductions on validation loss plateaus and early stopping. Validation results select both a balanced accuracy classification threshold and a high AD recall threshold before evaluation on the held-out test set.

The following figure shows the inference pipeline for a single MRI slice.

![Inference pipeline for a single MRI slice](./figures/inference_pipeline.jpg)

The following figure shows the training pipeline for a single MRI slice.

![Training pipeline for a single MRI slice](./figures/training_pipeline_figure.jpg)

(Training pipeline figure generated with GPT-5.6 Sol)

## Dataset structure

The supplied ADNI data is organised as three nested levels: patients own scans, and scans own slices.
Each slice is a 256x240 grayscale JPEG named `<scan id>_<slice index>.jpeg`, so `218391_78.jpeg` is slice 78 of scan `218391`, and that scan contributes exactly 20 consecutive slices numbered 78 to 97.
`meta_data_with_label.json` maps each scan id to its originating patient, for example scan `932015` belongs to patient `068_S_0473`, which is the only way to tell that two differently numbered scans come from the same person.
The supplied training set holds 1,076 scans (21,520 slices) from 565 patients, and the held-out test set holds 450 scans (9,000 slices) from 331 patients, split into AD and NC folders.

## Written Feasibility Report

The following is the written feasibility report completed at the midpoint of the project on the 13/10/2026. It outlines the basic acceptance criteria of the project and core concepts for the ConvNeXt model.

It is important to note that the basic shallow CNN smoketest completed within this feasibility report utilised a different input and output architecture compared with the final model. This was that it was a scan-level classifier that made predictions based on the average of 12 slices from a scan, rather than a slice-level classifier that makes predictions based on a single slice and its adjacent slices like the final model. The change to the slice-level classifier was made afterwards, with the driving reason being that at the beginning of the project, the model had trouble learning the training data and was underfitting. I thought a reason for this was that the model did not have enough training samples to learn from and thus the scan-level sample size of 1076 was increased to a slice-level sample size of 21,520.

## Feasibility Report: Preparing a ConvNeXt Model for Neuroimaging, AD vs NC Classification

### 1. User Need Scope & Acceptance Criteria

- **User Scope:** The intended users are medical researchers or healthcare professional who may use the model to assist them in diagnosing patients from the MRI scans.
- The prototype model will be used to determine if a ConvNext model is useful for the classification context, and if the heavier model provides value over a basic CNN classifier.
- **Acceptance Criteria:** Aim for 80% minimum test accuracy. Aim for High Recall at least 95%. Model can be trained within the hardware constraints set by the Rangpur HPC, peak VRAM < 40GB.

### 2. Model Choice & Core Concepts

- Dataset configuration will be used with the originally provided train/test split. Training set will be divided into an 80/20 training/validation set using the validation performance to tune hyperparameters of the model. Final inference will then be run on the test set to determine the true performance of the model.
- ConvNeXt (Difficulty Band Hard) model architecture will be used. Driving reason for this being that they are known to match or outperform conventional CNNs. They use large depth wise kernels, wide hidden layers, and less activation functions to mimic the behaviour of a transformer.
  - This leads to the purpose of the research task being if this heavier model has an improvement over conventional CNNS for the AD vs NC application.
- It is recommended that the model would be used in conjunction with professional opinion as this is a safety critical application. As a result:
  - We **aim for high recall** as a misclassified AD would be detrimental to the patient.

### 3. Preliminary Feasibility Evidence

- Dataset is provided as a separated training and test set with clear distinctions of patient numbers and MRI images in the format of &lt;patient_number&gt;\_&lt;image_number&gt;.
  - This can be used to ensure patient-level separation of the data.
- A smoke test with a shallow CNN was created to judge the effectiveness of a basic CNN model. The result was trained on my laptop on CPU, achieving a validation accuracy of 80% after 46 epochs completing training within the hour. Results are in the appendix.

### 4. Risk Budget and Fallback

Risks include:

- Patient-level data leakage. It is important to ensure that during training of the model that data is loaded correctly otherwise it is likely that the model will overfit on train set, provide poor inference on the test set or produce incorrect metrics.
  - To mitigate this risk it is important to view results carefully.
- Training Risk: Training of the heavier ConvNeXt model may take a long time so it is important that adequate time is provided to train the model on the Rangpur HPC. Adequate time must be allocated for model training, inference, performance evaluation and interpretation of results within the project deadline.
  - If training is incomplete/the model resulted in poor performance, the fallback is to still perform the evaluation and comparison to the basic CNN on the model that we ended up with.

### Appendix, Smoke Test Results:

![Smoke Test Confusion Matrix](./figures/smoke_test_confusion_matrix.png)

## Dependencies

The following tables lists the project dependencies.

| Package | Version | Required | Used for |
| --- | --- | --- | --- |
| Python | 3.11.15 | yes |  |
| PyTorch | 2.13.0 | yes | Deep learning framework: the model, training loop, data loading and GPU support |
| Pillow | 12.3.0 | yes | Image reading and writing |
| Matplotlib | 3.11.1 | optional | Result plots only |

Matplotlib is imported inside a `try`/`except ImportError` block in `save_training_curves`.
If it is missing, training still completes and every metric is still printed and written to `metrics.json` however the plots are skipped.

### Hardware

The reported results were produced on a single NVIDIA A100 on the Rangpur HPC.

## Running the project

On the course cluster, submit the training experiment  with:

```bash
sbatch train_convnext.sh
```

Weights, JSON metrics, confusion matrices and `training_curves.svg` are saved under `output/convnext_tiny_64px_2_5d_plateau`.
After training, example test predictions and a labelled image grid can be produced with:

```bash
python predict.py \
  --data-dir /home/groups/comp3710/ADNI/AD_NC \
  --metadata-path /home/groups/comp3710/ADNI/meta_data_with_label.json \
  --device cuda
```

To re-score the metrics of an existing checkpoint without training, add `--evaluate-only` to `train.py`.

## Example inputs and outputs

Running `predict.py` saves a labelled grid of test examples to `output/convnext_tiny_64px_2_5d_plateau/prediction_examples.png`.
A copy is reproduced below.

![Example test predictions with AD probabilities](./figures/prediction_examples.png)

Each panel shows the centre slice of one 2.5D input as the model sees it, after the foreground crop, the resize to 64x64 and per-channel foreground normalisation.
The previous and next slices occupy the other two channels and are not displayed.
The caption gives the slice identifier, the true diagnosis, the predicted AD probability, and the decision taken at each of the two validation-selected operating points.

An example of the matching console output is:

```
1003730_88: actual=AD, P(AD)=0.204, balanced=NC, 95% validation recall=AD
1003730_89: actual=AD, P(AD)=0.213, balanced=NC, 95% validation recall=AD
1003730_90: actual=AD, P(AD)=0.519, balanced=NC, 95% validation recall=AD
1003730_91: actual=AD, P(AD)=0.456, balanced=NC, 95% validation recall=AD
```

These eight examples are the first eight slices the test loader returns rather than a random sample, so they all belong to scan `1003730` and show consecutive slices 88 to 95 of the same brain.
They are informative for exactly that reason.
Every slice is truly AD, every predicted probability sits between 0.20 and 0.52, and the two operating points therefore disagree on all eight: the balanced-accuracy threshold calls them NC and the high-recall threshold calls them AD.
At the balanced operating point these are eight consecutive false negatives from one scan.

Two properties of the model are visible here.
Predictions are made independently per slice, yet the probabilities move smoothly across neighbouring slices (0.204, 0.213, 0.519, 0.456, 0.247, 0.237, 0.323, 0.325), which is the expected consequence of adjacent slices sharing most of their anatomy through the 2.5D input.
The probabilities also sit well below 0.5 for a genuinely AD scan, which is the behaviour the secondary high-sensitivity operating point exists to address: lowering the threshold recovers all eight at the cost of specificity elsewhere.

## Reproducibility

Two seeds control the run, and both are recorded in `metrics.json` alongside the full configuration.

| Flag | Default | Controls |
| --- | --- | --- |
| `--seed` | `42` | the patient-level train/validation split only (`_patient_stratified_split`) |
| `--training-seed` | `123` | model initialization, epoch sampling order, augmentation, MixUp and DataLoader shuffling |

Scan `1012896` is excluded before splitting because its slices contain no usable foreground.

## Superseded subject-level experiment

The first model averaged features from 12 slices to make one prediction per
subject. It stopped at epoch 24 with a best validation accuracy of 59.07%, well
below the 80% target, so this design was superseded. Its dataset, model and
training code remain under `superseded/`.

## Scan-safe slice classification

The active data loader groups filenames by patient before making the 80/20
training and validation split, ensuring repeated scans and neighbouring slices
cannot cross the split. Each input contains the previous, centre and next MRI
slices as three channels. ConvNeXt returns one prediction for the centre slice,
with no cross-slice averaging. Test metrics are calculated over all 9,000 test
slices.

The ConvNeXt-Tiny stages use depths `(3, 3, 9, 3)` and channels
`(96, 192, 384, 768)`. Its original 1000-class parameter count matches
TorchVision at 28,589,128 parameters.

## Training setup

AdamW uses batch size 8, a base learning rate of `0.0001` and weight decay of
`0.05`.
Training linearly warms up for five epochs. After warm-up, the learning rate is
halved by a validation-plateau scheduler with patience 5, down to a minimum of
`0.000001`. Training uses stochastic depth `0.1`, label smoothing `0.1`, MixUp
`0.4`, EMA weights and gradient clipping at `1.0`.

## Early stopping

Training saves the model when validation loss improves by at least `0.0001` and
stops after 20 consecutive epochs without that improvement. Counting begins
after epoch 10 so the warm-up and initial learning period cannot stop training
prematurely. Final test inference uses the best saved weights.

## Hyperparameter changes

- Increased the learning rate from `0.00005` to `0.0001` after the lower rate
  produced stable training but plateaued.
- Replaced the constant post-warm-up rate with validation-plateau reductions.
- Increased early-stopping patience from 5 to 20 epochs.
- Reduced the minimum validation-loss improvement from `0.001` to `0.0001`.

## MRI preprocessing

Each original 256×240 slice receives a constant 224×224 crop centred on its
non-black foreground, which is resized to 64×64. Each of the three adjacent
slices is normalized using its own foreground mean and standard deviation,
while black background pixels remain zero. Training adds mild rotation,
translation, scaling, gamma, intensity and noise augmentation using one shared
spatial transform across all three channels.

## Cosine scheduler experiment

`superseded/train_cosine.py` preserves a comparison run using the same model and
preprocessing. It warms up to `0.0001`, follows cosine annealing down to
`0.000001`, and completes all 100 epochs without early stopping. Its weights
and metrics use a separate `output/convnext_tiny_slice_level_cosine` folder.

Test metrics and the confusion matrix are printed and saved as JSON even when
Matplotlib is unavailable; only the optional PNG is skipped.

## Training diagnosis

The cosine run plateaued at 66.14% training accuracy and approximately 61%
validation accuracy. A separate memorization check reached 100% accuracy on
eight real slices within a few updates at both `0.0001` and `0.001`, confirming
that the model, gradients and optimizer work correctly. The full-data plateau
is therefore more likely caused by weak slice-level supervision: all slices
inherit the scan diagnosis even though an individual slice may contain limited
diagnostic information, and the 21,520 correlated slices represent only 1,076
scans. Further fixed-epoch cosine training is unlikely to resolve this plateau.

## Slice-subsampling pilot

`superseded/train_subsampled.py` keeps the original scan split (`seed=42`) but uses
training seed `123`. Each epoch randomly selects four distinct slices from
every training scan and reshuffles them, while validation and test inference
still use every slice. This reduces each training epoch from 17,220 correlated
slices to 3,444 and stores results separately under
`output/convnext_tiny_slice_subsampling`. It otherwise uses the same 100-epoch
cosine setup.

## Small-input pilot

`superseded/train_small_input.py` trains on all 17,220 slices with batch size 8. A check
on eight images found that both a central 64×64 crop and the complete brain
resized to 64×64 reached 100% training accuracy; the resized version learned
faster, so the pilot uses it. Full validation loss and accuracy are reported
after every epoch.

The original 224×224, batch-48 run made only about 359 optimizer updates per
epoch, compared with 2,153 at batch 8. The smaller input also removes fine
scanner-specific detail and makes each receptive field cover more of the
brain. A 96×96, batch-8 run later reached 100% training accuracy but
generalised worse than 64×64, suggesting that batch 8 improved optimization
while the lower resolution also acted as useful regularisation. Because image
size and batch size changed together initially, their effects were not fully
isolated.

## Regularised small-input experiment

The 64×64 run reached 100% training and 80.53% validation accuracy but
overfit. `superseded/train_regularized.py` adds stochastic depth `0.1` back, increases
weight decay to `0.05`, and adds label smoothing `0.1`. It saves the best
validation-accuracy checkpoint for final testing.

## 96×96 experiment

`superseded/train_regularized_96.py` keeps batch size 8 and the same regularisation while
increasing the resized input to 96×96. It uses `bfloat16` on the A100 and stops
immediately if a loss or gradient becomes non-finite.

## MRI augmentation experiment

`superseded/train_augmented.py` returns to 64×64 inputs and keeps ConvNeXt-Tiny unchanged.
Training slices receive small random rotations, translations, scaling,
intensity variation and Gaussian noise before normalization. Validation and
test preprocessing remains deterministic.

## Validation split finding

The split contains no repeated scans or images, but metadata showed that 109
patients have different scans in both training and validation. This overlap is
class-skewed: 95 of 101 validation AD scans are from patients seen during
training, compared with only 104 of 223 test AD scans. This likely inflated
validation accuracy to 88.70% while test accuracy reached 72.66% with 57.22%
AD recall. Future model selection should use a diagnosis-stratified,
patient-grouped validation split while retaining slice-level classification
and the supplied test set.

## Generalisation experiment

`superseded/train_generalized.py` keeps ConvNeXt-Tiny and 64×64 slice classification but
uses a diagnosis-stratified patient split, balances diagnoses and patients
within each training epoch, and normalizes each slice from its own brain
foreground. Training uses mild spatial and intensity augmentation without
flipping anatomical laterality. The AD threshold is selected using validation
balanced accuracy before test inference. Scan `1012896` is excluded before
splitting because its supplied slice `1012896_94.jpeg` has no usable
foreground. It ran all 100 cosine-scheduled epochs while saving the best
validation-loss weights.

## MixUp and EMA experiment

`superseded/train_mixup_ema.py` keeps the same slice-level model and patient split while
adding batch MixUp to reduce memorisation. Validation and checkpoint selection
use exponential moving-average model weights. Early stopping is retained and
the best validation-loss weights are used for test inference.

## 2.5D input experiment

The final `train.py` replaces the three duplicated grayscale channels with the
previous, centre and next slices from the same scan. The centre slice keeps its
original label and remains one test prediction, so no scan-level averaging is
introduced. The first and last slices repeat their nearest available edge
slice. Splitting still happens before triplets are created, preventing context
from crossing training, validation or scan boundaries. The model and remaining
MixUp/EMA settings are unchanged. The final run uses validation-plateau
learning-rate reductions instead of the slower fixed cosine schedule.

The experiment also reports a secondary high-sensitivity operating point. Its
threshold is selected on validation data to achieve at least 95% AD recall
while retaining as much specificity as possible, then applied unchanged to the
test set. This reflects a screening use case where missing a potential positive
case is more costly, but it is an experimental comparison rather than a
clinically validated diagnostic rule.

## References

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [TorchVision ConvNeXt source](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)
- [Official ConvNeXt training recipe](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md)
- [MixUp](https://openreview.net/pdf?id=r1Ddp1-Rb)
- [Adjacent-slice 2.5D MRI classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC13085064/)
