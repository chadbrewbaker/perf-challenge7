# perf-challenge7 <br> <br> Activation Functions for nanoGPT

This challenge is to improve the performance of training Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) on CPU with the Shakespeare data set by speeding up a critical subroutine - the activation function. 

Here are a few articles on activation functions:

[Activation Funcations of Nerual Networks Explained](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) 

[How to Choose the Right Activation Function for Neural Networks](https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c)

The Huggingface Transformers repository Python implementations: [tensorflow](https://github.com/huggingface/transformers/blob/main/src/transformers/activations_tf.py) - [pytorch](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)


One of the most popular is [Gaussian Linear Error Units](https://arxiv.org/abs/1606.08415) (GELU)
```python
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

 We will simplify tensor types to an array of floats.

```c
#include <unistd.h>
#include <math.h>

void slow_gelu_(float* in, float* out,  uint64_t length){
    uint64_t i;
    float x;
    for(i=0;i<length;i++){
        x = in[i];
        out[i] = 0.5 * x * (1.0 + tanh(sqrt(M_2_PI) * 
                           (x+ 0.044715 * x * x * x)));
    }
}

```
There are three challenges:
* fast_gelu_(float32_t* in, float32_t* out,  uint64_t length)
* fast_gelu_(bfloat16* in, bfloat16* out,  uint64_t length) 
* fast_gelu_(float16* in, float16* out,  uint64_t length)

The benchmark will give you an error score from the reference. During the challenge we will come up with what error is acceptable. You can assume the same range of inputs as running nanoGPT on the Shakespeare data set.

Bonus challenge: 

* Design your own activation function for training nanoGPT on the Shakespeare data set. Replace [this function](https://github.com/karpathy/nanoGPT/blob/ae3a8d5fdd3ddb8b13fab182723476523961e3ab/model.py#L19) with your custom activation function, describe how it effects the number of iterations required to train the model, and independently benchmark an optimized version as above - the floating point precision is your choice.

Rules:

* Please target [Denis' bare metal](https://easyperf.net/blog/2022/05/28/Performance-analysis-and-tuning-contest-6#target-configuration) or a hosted [GitHub worker](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources).
 

```bash

# Have python3 and git in your PATH
bash setup.sh
# Requires hyperfine installed
bash bench_inference.sh
```

## References

[Reiner Rottmann](https://github.com/rrottmann/anguita) has an implementation of "Speed Improvement of the Back-Propagation on Current Generation
Workstations" D. Anguita, G. Parodi and R. Zunino. Proceedings of the World Congress on Neural Networking, 1993.

[Mathworld tanh()](https://mathworld.wolfram.com/HyperbolicTangent.html)

[GNU tanh()](https://github.com/bminor/glibc/blob/master/sysdeps/ieee754/dbl-64/s_tanh.c)

[OpenLibm tanh()](https://github.com/JuliaMath/openlibm/blob/master/src/s_tanh.c)

[Intel tanh() intrinsics](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-short-vector-math-library-ops/intrinsics-for-trigonometric-operations/mm-tanh-ps-mm256-tanh-ps.html)

[SIMD Library roughTanh()](https://github.com/ermig1979/Simd/blob/c7208ea24c54721200dfe724dc5ca70521ca6ac8/src/Simd/SimdMath.h#L232)
