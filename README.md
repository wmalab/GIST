# GIST
Graph-based Neural Network Inference of Chromatin Spatial Structures using Hi-C data

---

## Environment
Create a conda *env* with installed:
```
> conda env create -f environment.yml
```

## Disclaimer

This repository is built for our experiments. A general package/interface will release later. 

## Overview of model 
![model](/figure/model.png)
## Usage

## Input configuration
```json
{   "id": // experiment name e.g. "train_v2_X",
    "cool_data_path": // path of Hi-C data cooler "",
    "cool_file": // Hi-C data name, e.g. "Rao2014-IMR90-MboI-allreps-filtered.10kb.cool", 
    "cell_name": //cell type name, e.g. "Rao2014-IMR90-MboI-allreps-filtered",
    "resolution": // resolution of Hi-C, e.g. 10000 (10kb),
    "all_chromosomes": // valid chromosomes, e.g. ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "X", "16", "17", "18", "19", "20", "21", "22"],
    "train_valid_chromosomes": // chromsomes in the training, e.g. ["X"],
    "parameter": {
        "graph":{
            "max_len": // maximum size of graph, e.g. 1000 (|V| <= 1000 nodes)
            "num_clusters": // number of clusters of edges, e.g. 9,
            "cutoff_clusters": // filter Hi-C data in percentile, e.g. {"low": 5.0, "high": 100}, 
            "cutoff_cluster": // input clusters of edges for the training, e.g. 7 (feed in 0-6)
        },
        "feature":{
            "in_dim": // input dimension e.g. 300,
            "out_dim": // output dimension, e.g. 30
        },
        "G3DM":{
            "iteration": // training epoch, e.g. 100,
            "num_heads": // number of heads, e.g. 40,
            "graph_dim": // dimensions of node, e.g.{"in_dim":30, "hidden_dim":10, "out_dim": 3}
        }
    }
}
```
## Citation