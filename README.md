# SMAL4V (SMAL for Videos)

This repository contains research done as part of the diploma thesis, which will be available for download at https://dspace.cvut.cz/ after the defence. The installation instruction and example results follow.

All the code from this repository is free to use if the original source is cited. Note that it has some dependences that are protected by different licences, including *SMAL (A Skinned Multi-Animal Linear Model of 3D Animal Shape)*. Consult https://smal.is.tue.mpg.de/license.html for the information on *SMAL* licensing.

This research ows a lot to the following works:

* [[model](https://smal.is.tue.mpg.de/)|[paper](https://files.is.tue.mpg.de/black/papers/smal_cvpr_2017.pdf)] *3D Menagerie: Modeling the 3D Shape and Pose of Animals* 
* [[code](https://github.com/benjiebob/WLDO)|[paper](https://arxiv.org/abs/2007.11110)] *Who Left the Dogs Out? 3D Animal Reconstruction with Expectation Maximization in the Loop* 
* [[code](https://github.com/silviazuffi/smalst)]|[paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/533/6034_after_pdfexpress.pdf)] *Three-D Safari: Learning to Estimate Zebra Pose, Shape, and Texture from Images "In the Wild"* 

## Description of the Approach

## Installation Instructions

To install *SMAL4V*, folow the steps described below. If you find any of them not reproducable, please open an issue on github or contact me at iegorval@gmail.com. Note that all the instructions below are tested only on Ubuntu and you would probably need some version of Linux to run the project.

0. Make sure you have correctly installed CUDA as described at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html.

1. Install conda environment from `environment.yml` and activate it:
    * `conda env create -f environment.yml`
    * `conda activate smal4v`
2. Install PyTorch with CUDA support. Note that the command below corresponds to the CUDA v11.1. In case you have different CUDA versions, consult https://pytorch.org/get-started/locally/. 
    * `pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html`
3. Install a PyTorch version of *Neural 3D Mesh Renderer*. Note that (as of time of writing) `pip` installation fails with this package. Also, you should change all `AT_CHECK` to `AT_ASSERT` in  `.cpp` files in `neural_renderer/cuda/` folder to be able to install it.
    *  `cd external`
    * `git clone https://github.com/daniilidis-group/neural_renderer` 
    * `cd neural_renderer`
    * `python setup.py install` 
4. Set the `PYTHONPATH` environment variable to include project root (or configure it in your IDE).
    * `export PYTHONPATH=[YOUR_PATH_TO_SMAL4V]`
    * `export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/home/viegorova/miniconda3/envs/smalmv/lib`
