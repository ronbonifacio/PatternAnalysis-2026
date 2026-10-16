# ADNI ConvNeXt-Tiny classifier

This project uses a layer-by-layer PyTorch implementation of ConvNeXt-Tiny to
classify ADNI MRI subjects as Alzheimer's disease (AD) or normal control (NC).
The architecture follows the official TorchVision implementation but does not
instantiate a prebuilt model or use pretrained weights.

Each subject contains multiple 2D MRI slices. The selected slices are passed
through the same ConvNeXt encoder independently, producing a 768-dimensional
feature vector for every slice. The vectors belonging to one subject are then
averaged and passed to a two-class linear classifier. This produces one AD/NC
prediction per subject rather than one prediction per slice.

TorchVision's standard ConvNeXt only performs spatial average pooling within
each image. This project retains that operation and adds a second averaging
step across the slices of each subject, matching the approach used by the basic
CNN classifier.

The local implementation uses the ConvNeXt-Tiny stage depths `(3, 3, 9, 3)`
and channel sizes `(96, 192, 384, 768)`. With the original 1000-class head, its
parameter count matches TorchVision ConvNeXt-Tiny at 28,589,128 parameters.

## References

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [TorchVision ConvNeXt source](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)
