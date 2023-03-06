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
