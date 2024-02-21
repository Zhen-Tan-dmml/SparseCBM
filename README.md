# SparseCBM (AAAI'24)

Datasets and baselines are available in our prevous work: [CBM-NLP](https://github.com/Zhen-Tan-dmml/CBM_NLP.git)

## Abstract

Large Language Models (LLMs) have achieved unprecedented breakthroughs in various natural language processing domains. However, the enigmatic ``black-box'' nature of LLMs remains a significant challenge for interpretability, hampering transparent and accountable applications. 

While past approaches, such as attention visualization, pivotal subnetwork extraction, and concept-based analyses, offer some insight, they often focus on either local or global explanations within a single dimension, occasionally falling short in providing comprehensive clarity. 

In response, we propose a novel methodology anchored in sparsity-guided techniques, aiming to provide a holistic interpretation of LLMs. 

Our framework, termed \textit{SparseCBM}, innovatively integrates sparsity to elucidate three intertwined layers of interpretation: input, subnetwork, and concept levels. In addition, the newly introduced dimension of interpretable inference-time intervention facilitates dynamic adjustments to the model during deployment. 

Through rigorous empirical evaluations on real-world datasets, we demonstrate that SparseCBM delivers a profound understanding of LLM behaviors, setting it apart in both interpreting and ameliorating model inaccuracies. Codes are provided in supplements.

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
