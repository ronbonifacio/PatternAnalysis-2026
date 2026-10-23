# ADNI ConvNeXt-Tiny classifier

This project uses a layer-by-layer PyTorch implementation of ConvNeXt-Tiny to
classify individual ADNI MRI slices as Alzheimer's disease (AD) or normal
control (NC). It follows the official TorchVision architecture without
instantiating a prebuilt model or using pretrained weights.

## Superseded subject-level experiment

The first model averaged features from 12 slices to make one prediction per
subject. It stopped at epoch 24 with a best validation accuracy of 59.07%, well
below the 80% target, so this design was superseded. Its dataset, model and
training code remain in the three `*_subject_level_superseded.py` files.

## Scan-safe slice classification

The active data loader groups filenames by MRI scan before making the 80/20
training and validation split. It then flattens each subset into individual
slices, ensuring neighbouring slices from one scan cannot cross the split.
ConvNeXt receives one 224×224 slice and returns one AD/NC prediction, with no
cross-slice averaging. Test accuracy and the confusion matrix are calculated
over the 9,000 test slices.

The ConvNeXt-Tiny stages use depths `(3, 3, 9, 3)` and channels
`(96, 192, 384, 768)`. Its original 1000-class parameter count matches
TorchVision at 28,589,128 parameters.

## Training setup

AdamW uses a base learning rate of `0.0001` and weight decay of `0.01`.
Training linearly warms up for five epochs. After warm-up, the learning rate is
halved by a validation-plateau scheduler with patience 5, down to a minimum of
`0.000001`. Stochastic depth is disabled and gradients are clipped to `1.0`.

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
non-black foreground. The fixed crop size removes empty border without
rescaling individual brains. Foreground mean and standard deviation are then
calculated from training scans only and reused to normalize training,
validation and test slices, while black background pixels remain zero.

## Cosine scheduler experiment

`train_cosine.py` provides a separate comparison run using the same model and
preprocessing. It warms up to `0.0001`, follows cosine annealing down to
`0.000001`, and completes all 100 epochs without early stopping. Its weights
and metrics use a separate `output/convnext_tiny_slice_level_cosine` folder.

Test metrics and the confusion matrix are printed and saved as JSON even when
Matplotlib is unavailable; only the optional PNG is skipped. A completed
cosine checkpoint can be tested without retraining with
`sbatch train_convnext_cosine.sh --evaluate-only`.

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

`train_subsampled.py` keeps the original scan split (`seed=42`) but uses
training seed `123`. Each epoch randomly selects four distinct slices from
every training scan and reshuffles them, while validation and test inference
still use every slice. This reduces each training epoch from 17,220 correlated
slices to 3,444 and stores results separately under
`output/convnext_tiny_slice_subsampling`.
It otherwise uses the same 100-epoch cosine setup and can be submitted with
`sbatch train_convnext_subsampled.sh`.

## Small-input pilot

`train_small_input.py` trains on all 17,220 slices with batch size 8. A check
on eight images found that both a central 64×64 crop and the complete brain
resized to 64×64 reached 100% training accuracy; the resized version learned
faster, so the pilot uses it. Full validation loss and accuracy are reported
after every epoch. Run it with `sbatch train_convnext_small_input.sh`.

## Regularised small-input experiment

The 64×64 run reached 100% training and 80.53% validation accuracy but
overfit. `train_regularized.py` adds stochastic depth `0.1` back, increases
weight decay to `0.05`, and adds label smoothing `0.1`. It saves the best
validation-accuracy checkpoint for final testing. Run it with
`sbatch train_convnext_regularized.sh`.

## 96×96 experiment

`train_regularized_96.py` keeps batch size 8 and the same regularisation while
increasing the resized input to 96×96. It uses `bfloat16` on the A100 and stops
immediately if a loss or gradient becomes non-finite. Run it with
`sbatch train_convnext_regularized_96.sh`.

## References

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [TorchVision ConvNeXt source](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)
