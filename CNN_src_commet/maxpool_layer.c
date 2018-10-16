#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

// 进行max pooling 前后的特征图通道数不变 所以l.c =l.out_c 
// get the output after maxpooling
image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride)
{
    fprintf(stderr, "Maxpool Layer: %d x %d x %d image, %d size, %d stride\n", h,w,c,size,stride);
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = (w-1)/stride + 1;
    l.out_h = (h-1)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));// 这里难道是要存起来 输出的data中 每一个样本 每一个通道 每一个位置的得到最大值的索引
    l.output =  calloc(output_size, sizeof(float));// output data
    l.delta =   calloc(output_size, sizeof(float));// output data change
    #ifdef GPU
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    int stride = l->stride;
    l->h = h;// input
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w-1)/stride + 1;// output
    l->out_h = (h-1)/stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int b,i,j,k,m,n;
    int w_offset = (-l.size-1)/2 + 1;// 在一个filter当中从最左边开始
    int h_offset = (-l.size-1)/2 + 1;

    int h = (l.h-1)/l.stride + 1;// 计算出输出的大小，说明pooling也是有重叠的
    int w = (l.w-1)/l.stride + 1;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));//输出的index
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;//找到最大的值的索引
                            max   = (val > max) ? val   : max;// 找到最大的值
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;// 里面是将所有的max的位置存起来，共h*w*c*l.batch个
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int i;
    int h = (l.h-1)/l.stride + 1;
    int w = (l.w-1)/l.stride + 1;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        // 明白为什么delta＋ 了 因为各个pooling部分有重叠
        state.delta[index] += l.delta[i];//state.delta对于不是最大的位置都为0，这个只update其中有max标记的位置
    }
}

