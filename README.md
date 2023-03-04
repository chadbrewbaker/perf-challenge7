# perf-challenge7 <br> <br> Tensor tanh()

This challenge is to improve the performance of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) on CPU with the Shakespeare data set by speeding up a critical subroutine - hyperbolic tangent of tensors. 

[Activation Funcations of Nerual Networks Explained](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) discusses why tanh() has useful properties as an activation function for neural networks.

Close only counts in horseshoes, hand granades, and Machine Learning. You can trade off tanh() accuracy for speed.

[https://github.com/rrottmann/anguita](Reiner Rottmann) has an implementation of "Speed Improvement of the Back-Propagation on Current Generation
Workstations" D. Anguita, G. Parodi and R. Zunino. Proceedings of the World Congress on Neural Networking, 1993.

Rules:

* Try to write your submission as if it were a pull request to nanoGPT and any of the underlying Python compute kernel libraries such as PyTorch. The goal is to democratize small transformer models.

* Please target [Denis' bare metal](https://easyperf.net/blog/2022/05/28/Performance-analysis-and-tuning-contest-6#target-configuration) or a hosted [GitHub worker](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources). 


```bash

# Have python3 and git in your PATH
bash setup.sh	
# Have hyperfine installed
bash bench_inference.sh
bash bench_token.sh
bash bench_training.sh
```
