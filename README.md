<!-- <p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222783-fdda535f-e132-4fdd-8871-2408cd29a264.png' width="50%">
</p> -->

# OpenGenome: A Comprehensive Benchmark of genomic foundation models

<!-- <p align="left">
<a href="https://arxiv.org/abs/2306.11249" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2306.11249-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/blob/master/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<!-- <a href="https://huggingface.co/OpenSTL" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-OpenSTL-blueviolet" /></a> -->
<!-- <a href="https://openstl.readthedocs.io/en/latest/" alt="docs">
    <img src="https://readthedocs.org/projects/openstl/badge/?version=latest" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/chengtan9907/SimVPv2?color=%23FF9600" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="resolution">
    <img src="https://img.shields.io/badge/issue%20resolution-1%20d-%23B7A800" /></a>
<a href="https://img.shields.io/github/stars/chengtan9907/OpenSTL" alt="arXiv">
    <img src="https://img.shields.io/github/stars/chengtan9907/OpenSTL" /></a>
</p> --> 

<!-- [📘Documentation](https://openstl.readthedocs.io/en/latest/) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](docs/en/model_zoos/video_benchmarks.md) |
[🤗Huggingface](https://huggingface.co/OpenSTL) |
[👀Visualization](docs/en/visualization/video_visualization.md) |
[🆕News](docs/en/changelog.md) -->

## Introduction

OpenGenome is a comprehensive benchmark for evaluating genomic foundation model, encompassing a broad spectrum of methods and diverse tasks, ranging from predicting gene location and function, identifying regulatory elements, and studying species evolution. OpenGenome offers a modular and extensible framework, excelling in user-friendliness, organization, and comprehensiveness. The codebase is organized into three abstracted layers, namely the core layer, algorithm layer, and user interface layer, arranged from the bottom to the top.

<p align="center" width="100%">
  <img src='/liuzicheng/ljh/hyena-dna/assets/architecture.png' width="90%">
</p>
<!-- <p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222226-61e6b8e8-959c-4bb3-a1cd-c994b423de3f.png' width="90%">
</p> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview

<!-- <details open>
<summary>Major Features and Plans</summary> -->

<!-- - **Flexiable Code Design.**
  OpenSTL decomposes STL algorithms into `methods` (training and prediction), `models` (network architectures), and `modules`, while providing unified experiment API. Users can develop their own STL algorithms with flexible training strategies and networks for different STL tasks.

- **Standard Benchmarks.**
  OpenGenome will support standard benchmarks of Genome foundation models with training and evaluation. We are working on training benchmarks and will update results synchronizingly.

- **Plans.**
  We plan to provide benchmarks of various Genomic foundation models and Meta Genomic foundation architectures in various genomic application tasks, e.g., predicting gene location and function, identifying regulatory elements, and studying species evolution etc. We encourage researchers interested in Genomic foundation models to contribute to OpenGenome or provide valuable advice! -->

</details>

<details open>
<summary>Code Structures</summary>

- `OpenGenome/configs` contains configuration for benchmark evaluation.
- `OpenGenome/data` contains datasets.
- `OpenGenome/notebook` contains analysis and visualization notebooks.
- `OpenGenome/src` contains source code for evaluation piplines.
- `OpenGenome/weight` contains pretrained weights for benchmark evaluation.
- `OpenGenome/experiment` contains scripts for experiment management.


</details>

## News and Updates

[2023-06-19] `OpenGenome` v0.3.0 is released and will be enhanced in [#25](https://github.com/chengtan9907/OpenSTL/issues/25).

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/chengtan9907/OpenSTL
cd OpenGenome
conda env create -f environment.yml
conda activate OpenGenome
python setup.py develop
```

<!-- <details close>
<summary>Dependencies</summary>

* argparse
* dask
* decord
* fvcore
* hickle
* lpips
* matplotlib
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* python<=3.10.8
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
</details>

Please refer to [install.md](docs/en/install.md) for more detailed instructions. -->

## Getting Started

Here is an example of single GPU non-distributed training HyenaDNA on demo_human_or_worm dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python train.py -m train experiment=hg38/genomic_benchmark_mamba \
        dataset.dataset_name=demo_human_or_worm \
        wandb.id=demo_human_or_worm_hyenadna \
        train.pretrained_model_path=path/to/pretrained_model \
        trainer.devices=1
```
## Repeat the experiment
Please see [experiment.MD](experiment/experiment.MD) for the details of experiment management.
<!-- ## Tutorial on using Custom Data

For the convenience of users, we provide a tutorial on how to train, evaluate, and visualize with OpenSTL on custom data. This tutorial enables users to quickly build their own projects using OpenSTL. For more details, please refer to the [`tutorial.ipynb`](examples/tutorial.ipynb) in the `examples/` directory.

We also provide a Colab demo of this tutorial:

<a href="https://colab.research.google.com/drive/19uShc-1uCcySrjrRP3peXf2RUNVzCjHh?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<p align="right">(<a href="#top">back to top</a>)</p> -->

## Overview of Model Zoo and Datasets

We support various Genomic foundation models and provide [benchmarks](https://github.com/chengtan9907/OpenSTL/tree/master/docs/en/model_zoos) on various STL datasets. We are working on add new methods and collecting experiment results.

* Spatiotemporal Prediction Methods.

    <details open>
    <summary>Currently supported methods</summary>

    - [x] [HyenaDNA](https://arxiv.org/abs/2306.15794) (NeurIPS'2023)
    - [x] [Caduceus](https://arxiv.org/abs/2403.03234) (Arxiv'2024)
    - [x] [DNABERT](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680) (Bioinformatics'2021)
    - [x] [DNABERT-2](https://arxiv.org/pdf/2306.15006.pdf) (Arxiv'2023)
    - [x] [The Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3.abstract) (BioRxiv'2023)
    - [x] [Gena-LM](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1) (BioRxiv'2023)
   

    

* Genomic foundation models Benchmarks ([prepare_data](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data) or [Baidu Cloud](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk)).

    <details open>
    <summary>Currently supported datasets</summary>

    - [x] [Genomic benchmark](https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8) (BMC Genomic Data'2023) [[download](https://sites.google.com/berkeley.edu/robotic-interaction-datasets)] [[config](configs/bair)]
    - [x] [GUE](https://arxiv.org/pdf/2306.15006.pdf) (Arxiv'2023) [[download](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view)] [[config](configs/human)]
    - [x] [Promoter prediction](https://basespace.illumina.com/projects/66029966/about) (BioRxiv'2023) [[download](https://www.csc.kth.se/cvap/actions/)] [[config](configs/kth)]
    - [x] [Splice site prediction](https://dl.acm.org/doi/10.1177/0278364913491297) (Cell Press'2019) [[download](https://basespace.illumina.com/projects/66029966/about)] [[config](configs/kitticaltech)]
    - [x] [Drosophila enhancer activity prediction](https://www.nature.com/articles/s41588-022-01048-5) (Nature Genetics'2022) [[download](https://data.starklab.org/almeida/DeepSTARR/Data/)] [[config](configs/kinetics)]
    - [x] [Genomic Structure Prediction](https://www.nature.com/articles/s41588-022-01065-4) (Nature Genetics'2022) [[download](https://github.com/jzhoulab/orca?tab=readme-ov-file)] [[config](configs/mmnist)]
    

    </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Visualization

We present visualization examples of HyenaDNA below. For more detailed information, please refer to the [visualization](docs/en/visualization/).

- For species classification task, visualization of t-sne embedding  can be found in [notebook/gene_cluster.ipynb](notebook/gene_cluster.ipynb). 

- For Genomic Structure Prediction, visualization of predicted structures and ground truth structures are shown in [assets/png](assets/png) after running the experiment.

<div align="center">

| DNA embedding clustering| 
<!-- | :---: | | -->
<div align=center><img src='assets/dna_clustering.png' height="auto" width="260" ></div> 


|Bulk RNA Expression|
 <div align=center><img src='notebook/Bulk_RNA_Expression.png' height="auto" width="260" ></div> 

|Drosophila_enhancers_prediction|
 <div align=center><img src="notebook/Drosophila_Enhancers_Prediction.png" height="auto" width="260" ></div> 


<!-- | Moving MNIST-CIFAR | KittiCaltech |
| :---: | :---: |
|  <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_ConvLSTM.gif' height="auto" width="260" ></div> |

| KTH | Human 3.6M | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_ConvLSTM.gif' height="auto" width="260" ></div> |

| Traffic - in flow | Traffic - out flow |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_in_flow_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_out_flow_ConvLSTM.gif' height="auto" width="260" ></div> |

| Weather - Temperature | Weather - Humidity |
|  :---: |  :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_ConvLSTM.gif' height="auto" width="360" ></div>|

| Weather - Latitude Wind | Weather - Cloud Cover | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_ConvLSTM.gif' height="auto" width="360" ></div> |

| BAIR Robot Pushing | Kinetics-400 | 
| :---: | :---: |
| <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872182-4f31928d-2ebc-4407-b2d4-1fe4a8da5837.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872560-00775edf-5773-478c-8836-f7aec461e209.gif' height="auto" width="260" ></div> | -->

</div>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

The framework of OpenGenome is insipred by [HyenaDNA](https://github.com/HazyResearch/hyena-dna)

<!-- ## Citation

If you are interested in our repository or our paper, please cite the following paper:

```

``` -->

## Contribution and Contact

For adding new features, looking for helps, or reporting bugs associated with `OpenGenome`, please open a [GitHub issue](https://github.com/chengtan9907/OpenSTL/issues) and [pull request](https://github.com/chengtan9907/OpenSTL/pulls) with the tag "new features", "help wanted", or "enhancement". Feel free to contact us through email if you have any questions.

- Jiahui Li (jiahuili.jimmy@gmail.com), Westlake University 


<p align="right">(<a href="#top">back to top</a>)</p>
