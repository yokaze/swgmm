# Sliced Wasserstein GMM

This repository contains an alternative implementation of "Sliced Wasserstein Distance for Learning Gaussian Mixture Models" proposed by Kolouri et., al.

<p align="center">
  <img src="https://github.com/yokaze/swgmm/blob/master/figures/swgmm.png" />
</p>

The figures describe the status of estimation (top-left), transport cost for each observation (middle-left), alignment of empirical and estimated distributions (bottom-left), and the history of estimated sliced Wasserstein distance (right).
You get this result in animation by running `swgmm.py`.

**NOTE:** This implementation does not estimate full covariance matrix, instead it only estimates the diagonal components.
Also, it applies gradients on the logarithm of unnormalized Gaussian weights and those of diagonal components in covariance matrices. These changes make the implementation easier, at the expense of moderate performance degradation.

## Resources

- Comparison of KL, Wasserstein-1, and Wasserstein-2 distances with respect to the coordinate of the left Gaussian component.
It can be seen that KL divergence is suffered from gradient vanishing and a local mimimum, while W1 and W2 distances successfully figure out the global minimum.

<p align="center">
  <img src="https://github.com/yokaze/swgmm/blob/master/figures/loss.png" />
</p>

- Illustration of optimal transport alignment between discrete and continuous distributions.

<p align="center">
  <img src="https://github.com/yokaze/swgmm/blob/master/figures/map.png" />
</p>

## Requirements

- TensorFlow 2.0
- R (to retrieve Old Faithful geyser dataset)
- No GPU is needed, this script runs on a CPU using three minutes

## Reference

- Soheil Kolouri, Gustavo K. Rohde, Heiko Hoffmann, "Sliced Wasserstein Distance for Learning Gaussian Mixture Models", in _Proc. CVPR_, 2018, pp. 3427--3436.<br />
  <span style="word-break: break-all;">
  https://arxiv.org/abs/1711.05376
  </span>
