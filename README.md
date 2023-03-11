# perf-challenge7 <br> <br> Gaussian Linear Error Units

This challenge is to improve the performance of [Gaussian Linear Error Units](https://arxiv.org/abs/1606.08415) (GELU)
which is a subroutine used by machine learning models as a neuron activation function. 


[Activation Funcations of Nerual Networks Explained](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) 


The Huggingface Transformers repository Python implementations: [tensorflow](https://github.com/huggingface/transformers/blob/main/src/transformers/activations_tf.py) - [pytorch](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)


Here is a Python implementation of GELU:

```python
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

Here is a naive implementation in C:

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
There are four challenges:
* fast_gelu_(float32_t* in, float32_t* out,  uint64_t length)
* fast_gelu_(bfloat16* in, bfloat16* out,  uint64_t length) 
* fast_gelu_(float16* in, float16* out,  uint64_t length)
* fn wasm_gelu(a: [wasm32::v128](https://doc.rust-lang.org/beta/core/arch/wasm32/struct.v128.html), b: wasm32::v128) -> wasm32::v128 //Assume float32 types

Rules:

* Please target [Denis' bare metal](https://easyperf.net/blog/2022/05/28/Performance-analysis-and-tuning-contest-6#target-configuration) or a hosted [GitHub worker](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources).
* As the contest progresses we will clarify acceptable GELU() error allowed.
* The range of float values will be sampled from [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) on the Shakespeare data set.
* For webassembly we will use the [Chromium runtime](https://chromium.googlesource.com/chromium/src/+/lkgr/headless/README.md). Here are the [wasm intinsics](https://doc.rust-lang.org/beta/core/arch/wasm32/index.html).
 

## References

[Reiner Rottmann](https://github.com/rrottmann/anguita) has an implementation of "Speed Improvement of the Back-Propagation on Current Generation
Workstations" D. Anguita, G. Parodi and R. Zunino. Proceedings of the World Congress on Neural Networking, 1993.

[Mathworld tanh()](https://mathworld.wolfram.com/HyperbolicTangent.html)

[GNU tanh()](https://github.com/bminor/glibc/blob/master/sysdeps/ieee754/dbl-64/s_tanh.c)

[OpenLibm tanh()](https://github.com/JuliaMath/openlibm/blob/master/src/s_tanh.c)

[Intel tanh() intrinsics](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-short-vector-math-library-ops/intrinsics-for-trigonometric-operations/mm-tanh-ps-mm256-tanh-ps.html)

[SIMD Library roughTanh()](https://github.com/ermig1979/Simd/blob/c7208ea24c54721200dfe724dc5ca70521ca6ac8/src/Simd/SimdMath.h#L232)
