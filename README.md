<div align='center'>

# Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent

</div>

## Abstract

Merging multiple expert models offers a promising approach for performing multi-task learning without accessing their original data. Existing methods attempt to alleviate task conflicts by sparsifying task vectors or promoting orthogonality among them. However, they overlook the fundamental requirement of model merging: ensuring the merged model performs comparably to task-specific models on respective tasks. We find these methods inevitably discard task-specific information that, while causing conflicts, is crucial for performance. Based on our findings, we frame model merging as a constrained optimization problem (i.e., minimizing the gap between the merged model and individual models, subject to the constraint of retaining shared knowledge) and solve it via adaptive projective gradient descent. Specifically, we align the merged model with individual models by decomposing and reconstituting the loss function, alleviating conflicts through data-free optimization of task vectors. To retain shared knowledge, we optimize this objective by projecting gradients within a shared subspace spanning all tasks. Moreover, we view merging coefficients as adaptive learning rates and propose a task-aware, training-free strategy. Experiments show that our plug-and-play approach consistently outperforms previous methods, achieving state-of-the-art results across diverse architectures and tasks in both vision and NLP domains.

## Installation

install the latest version in development

```bash
pip install -e . # install the package in editable mode
```

## Project Structure

The project is structured as follows:

- `fusion_bench/`: the main package of the benchmark.
  - `method`: contains the implementation of the fusion methods.
  - `modelpool`: contains the implementation of the model pool, responsible for managing the models and dataset to be loaded.
  - `taskpool`: contains the implementation of the task pool, responsible for evaluating the performance of models returned by the algorithm.
- `config/`: configuration files for the benchmark. We use [Hydra](https://hydra.cc/) to manage the configurations.
  - `method`: configuration files for the fusion methods.
    
  - `modelpool`: configuration files for the model pool.
  - `taskpool`: configuration files for the task pool.
  - `model`: configuration files for the models.
  - `dataset`: configuration files for the datasets.

## How to run the experiments

 - For DOGE TA, run the following command:

     ```shell
     fusion_bench \
    method=DOGE_TA/DOGE_TA \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
     ```
- For DOGE AM, run the following command:

     ```shell
     fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging_doge_ta \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
     ```
