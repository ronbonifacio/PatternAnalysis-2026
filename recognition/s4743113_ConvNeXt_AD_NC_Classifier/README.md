# ADNI ConvNeXt-Tiny classifier

This project uses a layer-by-layer PyTorch implementation of ConvNeXt-Tiny to
classify ADNI MRI slices as Alzheimer's disease (AD) or normal control (NC).
The model uses adjacent slices as a 2.5D input and follows the official
TorchVision architecture without instantiating a prebuilt model or using
pretrained weights.

## Running the project

On the course cluster, submit the final experiment with:

```bash
sbatch train_convnext.sh
```

Weights, JSON metrics, confusion matrices and `training_curves.svg` are saved
under `output/convnext_tiny_64px_2_5d_plateau`. After training, example test
predictions and a labelled image grid can be produced with:

```bash
python predict.py \
  --data-dir /home/groups/comp3710/ADNI/AD_NC \
  --metadata-path /home/groups/comp3710/ADNI/meta_data_with_label.json \
  --device cuda
```

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
