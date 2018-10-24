# Gan_Architecture
create an architecture for Generative Adversarial Networks.
implementations 
## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)   

## Installation
    $ git clone https://github.com/VitoRazor/Gan_Architecture.git
    $ cd Gan_Architecture-master/
    $ pip install keras

## Implementations   
### GAN with SN
Implementation of Generative Adversarial Network with Spectral Normalization for Wasserstein-divergence 

[Code](myGan_w_sn.py)

Reference Paper:

Spectral normalization for generative adversarial networks:https://arxiv.org/abs/1802.05957

Wasserstein GAN: https://arxiv.org/abs/1701.07875

Result:

### GAN with info
Implementation of Generative Adversarial Network with InfoGAN and ACGAN, simultaneously using Spectral Normalization for Wasserstein-divergence.

[Code](myGan_info.py)

Reference Paper:

Auxiliary Classifier Generative Adversarial Network: https://arxiv.org/abs/1610.09585

Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets: https://arxiv.org/abs/1606.03657                                             
                 
Result:
<p align="left">
    <img src="http://eriklindernoren.se/images/aae.png" width="640"\>
</p>

