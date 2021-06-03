This repository contains the implementation for the paper *Information Theoretic StructuredGenerative Modeling*,

Specially thanks for the open-source codes shared by [*sagelywizard/pytorch-mdn*](https://github.com/sagelywizard/pytorch-mdn) and [*PyTorch-GAN*](https://github.com/eriklindernoren/PyTorch-GAN)

### Main Requirements

* [*Pytorch*](https://github.com/pytorch/pytorch)
* A GPU Machine

### Usage

#### The main experiments in the paper are put in the notebook format. 

#### Each file can be run independently
- GAN Example
- MINE Example
- Density Estimation
- Conditonal Estimation

#### To run ood_visualization.ipynb, please download the pretrained model in the ./model/ folder.

#### The other baselines can be run by calling
```shell
python MDN.py
python CGAN.py
```

#### GMM_VBGMM_CE.py provides codes for producing conditional CE for any mixture models obtained from scipy. 

Simply calling
```shell
compute_conditionalCE(joint, gm_joint)
```
in python to obtain the value, where joint should be bs*K and gm_joint is the class obtained from scipy.
