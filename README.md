# VisionDatasets.jl

Download and load common computer vision datasets for different tasks.

Datasets are grouped by task, so far including:

- classification

This package uses the dataset interface of [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl), meaning that every dataset defines:

- `LearnBase.getobs(ds, idx|idxs)` to load one or more samples
- `LearnBase.nobs(ds)` to get the number of samples in a dataset


Samples are represented as `Dict`s, and not as tuples of `(x, y)` because there is often more information associated with each sample, like the dataset split is part of.

The `Dict`s keys differ for every task, see below for more info.

## Install

```julia
]add https://www.github.com/lorenzoh/VisionDatasets.jl
```

## Usage

```julia
using VisionDatasets
using LearnBase: getobs, nobs

# load dataset files and labels
> dataset = ImageWoof.ImageWoof2_160()
ClassificationDataset(...)

# grab a sample
> getobs(dataset, 1)
Dict(:image => ..., :label => ..., ...)
```

## Datasets

### Classification 

A sample of a `ClassificationDataset` is a `Dict` with the following keys:

- `:image`: The image
- `:label`: the sample's label id, from 1 to number of classes; check `dataset.names` to access to corresponding label names
- `:split`: dataset split the sample is part of, e.g. `:train` or `:val`

#### Available classification datasets

- [ImageWoof](https://github.com/fastai/imagenette)
    - ImageWoof2 (160px)
    - ImageWoof2 (320px)
- [ImageNette](https://github.com/fastai/imagenette)
    - ImageNette2 (160 px)
    - ImageNette2 (320 px)