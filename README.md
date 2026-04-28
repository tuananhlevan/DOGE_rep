<div align='center'>

# [ICML 2025] Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent

</div>

## Abstract

Merging multiple expert models offers a promising approach for performing multi-task learning without accessing their original data. Existing methods attempt to alleviate task conflicts by sparsifying task vectors or promoting orthogonality among them. However, they overlook the fundamental target of model merging: the merged model performs as closely as possible to task-specific models on respective tasks. We find these methods inevitably discard task-specific information that, while causing conflicts, is crucial for performance. Based on our findings, we frame model merging as a constrained optimization problem (\ie, minimizing the gap between the merged model and individual models, subject to the constraint of retaining shared knowledge) and solve it via adaptive projective gradient descent. Specifically, we align the merged model with individual models by decomposing and reconstituting the loss function, alleviating conflicts through \textit{data-free} optimization of task vectors. To retain shared knowledge, we optimize this objective by projecting gradients within a \textit{shared subspace} spanning all tasks. Moreover, we view merging coefficients as adaptive learning rates and propose a task-aware, training-free strategy. Experiments show that our plug-and-play approach consistently outperforms previous methods, achieving state-of-the-art results across diverse architectures and tasks in both vision and NLP domains.

## Installation

install the latest version in development

```bash
conda create -n DOGE python==3.10 -y
conda activate DOGE
pip install -e . # install the package in editable mode
pip install transformers==5.0.0
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Change to your fitting cuda version
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

 - For DOGE TA ViT-L-14 7 tasks, run the following command:

     ```shell
     bash doge_ta.sh
     ```
- For Fisher Merging ViT-L-14 7 tasks, run the following command:

     ```shell
    bash fisher.sh
     ```
## Citation
If you find DOGE useful for your research and applications, please cite using this BibTeX:
```bash
@article{wei2025modeling,
  title={Modeling multi-task model merging as adaptive projective gradient descent},
  author={Wei, Yongxian and Tang, Anke and Shen, Li and Yuan, Chun and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2501.01230},
  year={2025}
}
