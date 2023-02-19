# perf-challenge7

This challenge is to improve the performance of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) on CPU with the Shakespeare data set. 

There are three benchmarks:
 
* Tokenizing the works of Shakespeare.

* Training the nanoGPT model for 100 iterations.

* Running inference on the nanoGPT Shakespare model with default settings.

[SetFit](https://github.com/huggingface/setfit) might provide inspiration on using integer encodings to improve performance.

Karpathy made a 25% speedup with [prime factorization tuning](https://twitter.com/karpathy/status/1621578354024677377). 

Rules:

* Try to write your submission as if it were a pull request to nanoGPT and any of the underlying Python compute kernel libraries. The goal is to democratize small transformer models.

* Please target [Denis' bare metal](https://easyperf.net/blog/2022/05/28/Performance-analysis-and-tuning-contest-6#target-configuration) or a hosted [GitHub worker](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources). 

* If you do find some interesting optimzations on GPU, please make sure they work under [Google Colab](https://colab.research.google.com) or [Intel ARC under WSL2](https://medium.com/intel-analytics-software/stable-diffusion-with-intel-arc-gpus-f2986bba8365).

Bonus:

* For the inference benchmark - see if you can get it running as [WASM](https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/browsers.html). Democratizing GPTs means allowing anyone to self host.  

Setup:

```bash

# Have python3 and git in your PATH
bash setup.sh	
# Have hyperfine installed
bash bench_inference.sh
bash bench_token.sh
bash bench_training.sh
```
