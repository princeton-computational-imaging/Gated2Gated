# Gated2Gated : Self-Supervised Depth Estimation from Gated Images

![architecture](assets/imgs/architecture.png)

This repository contains code for [Gated2Gated : Self-Supervised Depth Estimation from Gated Images](https://arxiv.org/pdf/2112.02416.pdf). Dataset will also be published soon.

## Summary
Gated cameras hold promise as an alternative to scanning LiDAR sensors with high-resolution 3D depth that is robust to back-scatter in fog, snow, and rain. Instead of sequentially scanning a scene and directly recording depth via the photon time-of-flight, as in pulsed LiDAR sensors, gated imagers encode depth in the relative intensity of a handful of gated slices, captured at megapixel resolution. Although existing methods have shown that it is possible to decode high-resolution depth from such measurements, these methods require synchronized and calibrated LiDAR to supervise the gated depth decoder -- prohibiting fast adoption across geographies, training on large unpaired datasets, and exploring alternative applications outside of automotive use cases. In this work, we fill this gap and propose an entirely self-supervised depth estimation method that uses gated intensity profiles and temporal consistency as a training signal. The proposed model is trained end-to-end from gated video sequences, does not require LiDAR or RGB data, and learns to estimate absolute depth values. We take gated slices as input and disentangle the estimation of the scene albedo, depth, and ambient light, which are then used to learn to reconstruct the input slices through a cyclic loss. We rely on temporal consistency between a given frame and neighboring gated slices to estimate depth in regions with shadows and reflections. We experimentally validate that the proposed approach outperforms existing supervised and self-supervised depth estimation methods based on monocular RGB and stereo images, as well as supervised methods based on gated images.

## Getting started
To get started, first clone this repository in your local directory using 

```
https://github.com/princeton-computational-imaging/Gated2Gated
```
For getting all the necessary packages, get the anaconda environment using:
```
conda env create -f environment.yml
```
Activate the environment using
```
conda activate gated2gated
```
<!-- Download the dataset once the link is available in `data` directory. -->
Download the dataset in the `data` directory.

## Quick Example
Infer depth for single example using
```
sh scripts/inference.sh
```
## Training
After downloading the pre-trained weights (from lower resolution, read [here](weights/initialization/README.md)), start training using:

```
sh scripts/train.sh
```

<!-- ## Evaluation
For downloading the final weights, please refer to [here](weights/final/README.md).

**TBC** -->

<!-- ## Additional Material -->

### Pre-trained Models
Our final model weights are available to download using this [link](https://drive.google.com/drive/folders/1iQlPkX_sz8SV6lTJDgcNRQPPyOGfO7SX?usp=sharing). More details can be found [here](weights/final/README.md).

### Results

![architecture](assets/imgs/albedo_ambient_examples.png)
![architecture](assets/imgs/cbar.png)
#### Day
![architecture](assets/gifs/day.gif)

#### Night
![architecture](assets/gifs/night.gif)

#### Fog
![architecture](assets/gifs/fog.gif)

#### Snow
![architecture](assets/gifs/snow.gif)

## Reference
If you find our work on gated depth estimation useful in your research, please consider citing our paper:

```bib
@misc{walia2021gated2gated,
      title={Gated2Gated: Self-Supervised Depth Estimation from Gated Images}, 
      author={Amanpreet Walia and Stefanie Walz and Mario Bijelic and Fahim Mannan and Frank Julca-Aguilar and Michael Langer and Werner Ritter and Felix Heide},
      year={2021},
      eprint={2112.02416},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This code in parts is inspired/borrowed from [monodepth2](https://github.com/nianticlabs/monodepth2) and [packnet-sfm](https://github.com/TRI-ML/packnet-sfm).

