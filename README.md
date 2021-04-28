# Deep-Polarimetric-HDR-Reconstruction
Deep HDR Reconstruction using Polarimetric Cues from the Polarization Camera

## Requirements
* Python 3.8.5
* Pytorch 1.2.0
* torchvision 0.4.0
* OpenCV 4.1.2.30
* Numpy 1.19.1
* Pillow 6.1.0
* OpenEXR 1.3.2
* CUDA 10.2
* cuDNN 7.6.5
* NVIDIA GTX 1080 TI

```
pip install -r requirements.txt
```

## Usage
### Dataset
The collected EdPolCommunity dataset can be found in the dataset folder on Google Drive: [pol_outdoor1](https://drive.google.com/file/d/18nhczTSCFMB4_oUZZzyF_kHhqNCt8MGs/view?usp=sharing), [pol_outdoor2](https://drive.google.com/file/d/1za16n_CeqPrNUAkFdxjT2Hf_bTB3cthi/view?usp=sharing)

### Pretrained model
The pretrained model checkpoints can be found in the checkpoints folder on [Google Drive](https://drive.google.com/file/d/1luFzTFl1top5VSuZWwZz676xugn7WKf_/view?usp=sharing)

### Inference
Sample code for inference using the DPHR model
```
python sample_code.py --i0 PATH/TO/POL0 --i45 PATH/TO/POL45 --i90 PATH/TO/POL90 --i135 PATH/TO/POL135 --out_dir PATH/TO/OUT/im_{idx}.png --weights model/DPHR_checkpoint.pt 
```
Training code will be added soon.

### Supplementary materials
We provide more visual comparisons in the supplementary material PDF. Namely, we provide more qualitative results on: 1) Evaluation of the HDR formulation. 2) Comparison with state-of-the-art methods.

## Reference
If you find this work useful in your research, please cite:
