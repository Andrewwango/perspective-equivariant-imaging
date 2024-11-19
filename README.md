# Perspective-Equivariance for Unsupervised Imaging with Camera Geometry

✨Accepted at ECCV 2024 [2nd Workshop on Traditional Computer Vision in the Age of Deep Learning](https://sites.google.com/view/tradicv/home?authuser=0).

[![arXiv](https://img.shields.io/badge/arXiv-2403.09327-green.svg)](https://arxiv.org/abs/2403.09327)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Andrewwango/perspective-equivariant-imaging)
[Demo webpage](https://andrewwango.github.io/perspective-equivariant-imaging) | [Demo notebook](demo.ipynb)

**Authors**: Andrew Wang, Mike Davies, School of Engineering, University of Edinburgh

**Abstract**: Ill-posed image reconstruction problems appear in many scenarios such as remote sensing, where obtaining high quality images is crucial for environmental monitoring, disaster management and urban planning. Deep learning has seen great success in overcoming the limitations of traditional methods. However, these inverse problems rarely come with ground truth data, highlighting the importance of unsupervised learning from partial and noisy measurements alone. We propose _perspective-equivariant imaging_ (EI), a framework that leverages classical projective camera geometry in optical imaging systems, such as satellites or handheld cameras, to recover information lost in ill-posed camera imaging problems. We show that our much richer non-linear class of group transforms, derived from camera geometry, generalises previous EI work and is an excellent prior for satellite and urban image data. Perspective-EI achieves state-of-the-art results in multispectral pansharpening, outperforming other unsupervised methods in the literature.

**Citation**
```
@article{wang2024perspective,
  title={Perspective-Equivariance for Unsupervised Imaging with Camera Geometry},
  author={Wang, Andrew and Davies, Mike},
  year={2024},
  url={https://arxiv.org/abs/2403.09327}
}
```

## 1. Results

![](img/eval_spacenet_pansharpen_noiseless.png)
