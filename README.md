# SparseCBM (AAAI'24)

## Install

We follow installation instructions from the [CEBaB](https://github.com/CEBaBing/CEBaB.git) repository, which mainly depends on [Huggingface](https://github.com/huggingface/transformers.git).

## Experiments

The code is tested on NVIDIA 3090 and A100 40/80GB GPU. An example for running the experiments is as follows:

```shell
bash run.sh
```

Note: It seems the random seed cannot control the randomness in parameter initialization in transformer, we suggest to run the code multiple times to get good scores.

## Citation
```
@article{tan2023sparsity,
  title={Sparsity-Guided Holistic Explanation for LLMs with Interpretable Inference-Time Intervention},
  author={Tan, Zhen and Chen, Tianlong and Zhang, Zhenyu and Liu, Huan},
  journal={arXiv preprint arXiv:2312.15033},
  year={2023}
}
