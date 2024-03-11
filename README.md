# Perspective-Equivariant Imaging: an Unsupervised Framework for Multispectral Pansharpening

[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-<COLOR>.svg)](https://arxiv.org/abs/<INDEX>)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Andrewwango/perspective-equivariant-imaging)
[Webpage](https://andrewwango.github.io/perspective-equivariant-imaging) | [Notebook](demo.ipynb)

**Authors**: Andrew Wang, Mike Davies, School of Engineering, University of Edinburgh

**Abstract**: Ill-posed image reconstruction problems appear in many scenarios such as remote sensing, where obtaining high quality images is crucial for environmental monitoring, disaster management and urban planning. Deep learning has seen great success in overcoming the limitations of traditional methods. However, these inverse problems rarely come with ground truth data, highlighting the importance of unsupervised learning from partial and noisy measurements alone. We propose _perspective-equivariant imaging_ (EI), a framework that leverages perspective variability in optical camera-based imaging systems, such as satellites or handheld cameras, to recover information lost in ill-posed optical camera imaging problems. This extends previous EI work to include a much richer non-linear class of group transforms and is shown to be an excellent prior for satellite and urban image data, where perspective-EI achieves state-of-the-art results in multispectral pansharpening, outperforming other unsupervised methods in the literature.

**Citation**
```
@article{wang2024perspective,
  title={Perspective-Equivariant Imaging: an Unsupervised Framework for Multispectral Pansharpening},
  author={Wang, Andrew and Davies, Mike},
  year={2024},
  url={https://arxiv.org/abs/<INDEX>}
}
```

## 1. Results

![](img/eval_spacenet_pansharpen_noiseless.png)