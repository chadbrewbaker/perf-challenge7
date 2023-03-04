# perf-challenge7 <br> <br> Tensor tanh()

This challenge is to improve the performance of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) on CPU with the Shakespeare data set by speeding up a critical subroutine - hyperbolic tangent of tensors. 

[Mathworld](https://mathworld.wolfram.com/HyperbolicTangent.html) describes tanh() in depth.

[Activation Funcations of Nerual Networks Explained](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) discusses why tanh() has useful properties as an activation function for neural networks.

You can trade off tanh() accuracy for speed. Close enough counts in horseshoes and Machine Learning. 

[Reiner Rottmann](https://github.com/rrottmann/anguita) has an implementation of "Speed Improvement of the Back-Propagation on Current Generation
Workstations" D. Anguita, G. Parodi and R. Zunino. Proceedings of the World Congress on Neural Networking, 1993.

[GNU tanh()](https://github.com/bminor/glibc/blob/master/sysdeps/ieee754/dbl-64/s_tanh.c)
[OpenLibm tanh()](https://github.com/JuliaMath/openlibm/blob/master/src/s_tanh.c)

[Intel tanh() intrinsics](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-short-vector-math-library-ops/intrinsics-for-trigonometric-operations/mm-tanh-ps-mm256-tanh-ps.html)


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
