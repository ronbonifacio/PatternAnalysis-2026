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

## Subject-safe slice classification

The active data loader groups filenames by subject before making the 80/20
training and validation split. It then flattens each subset into individual
slices, ensuring similar slices from one subject cannot cross the split.
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

## References

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [TorchVision ConvNeXt source](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)
