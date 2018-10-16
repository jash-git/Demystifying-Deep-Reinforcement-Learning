#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

// binary filter and filter 交换
void swap_binary(convolutional_layer *l)
{
    float *swap = l->filters;
    l->filters = l->binary_filters;
    l->binary_filters = swap;

    #ifdef GPU
    swap = l->filters_gpu;
    l->filters_gpu = l->binary_filters_gpu;
    l->binary_filters_gpu = swap;
    #endif
}
//
void binarize_filters2(float *filters, int n, int size, char *binary, float *scales)
{
    int i, k, f;
    // for 其中的n个filter，每一个filter 大小为size（不是size＊size大小）
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(filters[f*size + i]);
        }
        mean = mean / size;//求出这个filter的均值
        scales[f] = mean;
        for(i = 0; i < size/8; ++i){
            binary[f*size + i] = (filters[f*size + i] > 0) ? 1 : 0;
            for(k = 0; k < 8; ++k){
            }
        }
    }
}

// for each filter compute the mean, and change its value to mean and -mean
// 这里 n是feature map的个数， size是每一个 filter的 大小
void binarize_filters(float *filters, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(filters[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (filters[f*size + i] > 0) ? mean : -mean;//why compare to 0,not to mean??
        }
    }
}

//计算经过这个卷积层以后输出的特征图的 高度，和步长 stride还有卷积核的size有关
int convolutional_out_height(convolutional_layer l)
{
    int h = l.h;
    if (!l.pad) h -= l.size;
    else h -= 1;
    return h/l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    int w = l.w;
    if (!l.pad) w -= l.size;
    else w -= 1;
    return w/l.stride + 1;//stride 是啥？步长么？？
}

image get_convolutional_image(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}
// batch 样本个数，n 是outputs
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;// 不同样本中同一位置的值 加起来，其实Normalize的时候，比例系数是有除以 batch的个数的
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalize, int binary)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;// output numbers of feature map, that out_c
    l.binary = binary;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;
    l.batch_normalize = batch_normalize;

    l.filters = calloc(c*n*size*size, sizeof(float));
    l.filter_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));// c个filter卷积出一个值

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.filters[i] = scale*rand_uniform(-1, 1);//随机初始化filter的值
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    //输入数据通过 im2col_cpu操作变成了out_h*out_w*size*size*c
    l.col_image = calloc(out_h*out_w*size*size*c, sizeof(float));
    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch*out_h * out_w * n, sizeof(float));

    if(binary){
        l.binary_filters = calloc(c*n*size*size, sizeof(float));
        l.cfilters = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.filters_gpu = cuda_make_array(l.filters, c*n*size*size);
    l.filter_updates_gpu = cuda_make_array(l.filter_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.scales_gpu = cuda_make_array(l.scales, n);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

    l.col_image_gpu = cuda_make_array(l.col_image, out_h*out_w*size*size*c);
    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

    if(binary){
        l.binary_filters_gpu = cuda_make_array(l.filters, c*n*size*size);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, n);
        l.variance_gpu = cuda_make_array(l.variance, n);

        l.rolling_mean_gpu = cuda_make_array(l.mean, n);
        l.rolling_variance_gpu = cuda_make_array(l.variance, n);

        l.mean_delta_gpu = cuda_make_array(l.mean, n);
        l.variance_delta_gpu = cuda_make_array(l.variance, n);

        l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    }
#endif
    l.activation = activation;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    //对其中的n个filter
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        //对每一个filter的point进行尺度
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.filters[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
    }
}

void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    network_state state = {0};
    state.input = data;
    forward_convolutional_layer(l, state);
}
// 根据输入的Feature Map的大小调整网络的参数 （其实也就是调整了输出的network feature map 的大小而已）
void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->col_image = realloc(l->col_image,
            out_h*out_w*l->size*l->size*l->c*sizeof(float));// change ptr size :iker
    l->output = realloc(l->output,
            l->batch*out_h * out_w * l->n*sizeof(float));
    l->delta  = realloc(l->delta,
            l->batch*out_h * out_w * l->n*sizeof(float));

#ifdef GPU
    cuda_free(l->col_image_gpu);
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->col_image_gpu = cuda_make_array(l->col_image, out_h*out_w*l->size*l->size*l->c);
    l->delta_gpu =     cuda_make_array(l->delta, l->batch*out_h*out_w*l->n);
    l->output_gpu =    cuda_make_array(l->output, l->batch*out_h*out_w*l->n);
#endif
}
// for 这一批样本的 n个feature maps加偏置
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}
// for 这一批样本（batch个）的 n个feature maps的 尺度变换
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

// n feature map的个数， size 是feature map的大小
 // batch是样本的个数
//更新每一个bias 的值，通过原来每一张Feature Maps的bias_update值加上 delta里面的值的方式
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);//sum each feaure map of each sample,整张feature map有一个偏执
        }
    }
}
// 前向传播根据state当中的输入input，进行卷积操作，得到输出结果output
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    //l.outputs = l.out_h * l.out_w * l.out_c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);// fill (initial) output with 0 (batch 个样本)
    /*
    if(l.binary){
        binarize_filters(l.filters, l.n, l.c*l.size*l.size, l.binary_filters);
        binarize_filters2(l.filters, l.n, l.c*l.size*l.size, l.cfilters, l.scales);
        swap_binary(&l);
    }
    */

    if(l.binary){
        int m = l.n;
        int k = l.size*l.size*l.c;// 每个filter的大小 （这里把 c个通道的filter看作是一个）
        int n = out_h*out_w;

        char  *a = l.cfilters;
        float *b = l.col_image;
        float *c = l.output;//c 指向 output


        //for each sample
        for(i = 0; i < l.batch; ++i){
            im2col_cpu(state.input, l.c, l.h, l.w, 
                    l.size, l.stride, l.pad, b);// b is the result
            gemm_bin(m,n,k,1,a,k,b,n,c,n);
            c += n*m;// c是要输出的数据的指针，每一个feature map的头指针
            state.input += l.c*l.h*l.w;
        }
        scale_bias(l.output, l.scales, l.batch, l.n, out_h*out_w);
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);// 给output加上偏置
        activate_array(l.output, m*n*l.batch, l.activation);// pass active function(for 这一批样本的 n个输出特征图 的out_h*out_w个点进行激活处理)
        return;
    }

//    以下的操作是当filter不是二进制的时候

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;

    float *a = l.filters;
    float *b = l.col_image;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        //generate a Cool data structure for convolution.
        im2col_cpu(state.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        //以下的 0 0参数确定了第一种方式，总共有00 01 10 11四种
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        //指针指向下一块数据区域，存储下一个样本 ，input的大小也增加一倍
        c += n*m;
        state.input += l.c*l.h*l.w;//输入的数据位置，  将指针指到下一个样本input的位置
    }

    if(l.batch_normalize){
        if(state.train){
            mean_cpu(l.output, l.batch, l.n, l.out_h*l.out_w, l.mean);//输出 每一个feature map的mean。  
            variance_cpu(l.output, l.mean, l.batch, l.n, l.out_h*l.out_w, l.variance);// 根据mean 输出variance   
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.n, l.out_h*l.out_w);   
        } else {
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.n, l.out_h*l.out_w);
        }
        scale_bias(l.output, l.scales, l.batch, l.n, out_h*out_w);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
}

// 其实反向传播就做两个事情，都是通过梯度来做的。一个是计算出偏置的更新量，另一个计算出filter的更新量
void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;//feature map Number= filter Number
    int n = l.size*l.size*l.c;// filter size
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);// feature map size

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);//通过delta 来计算bias更新值 （每一张 feature map有一个 bias）

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;//for each sample's feature maps
        float *b = l.col_image;
        float *c = l.filter_updates;

        float *im = state.input+i*l.c*l.h*l.w;//单个输入的样本数据

        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);// 对经过求导后的数据准备 卷积，因此构造符合卷积的数据结构
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n); //对C的处理的结果就是放在了l.filter_updates当中

        // 如果有state.delta的值还是要更新
        if(state.delta){
            a = l.filters;
            b = l.delta + i*m*k;
            c = l.col_image;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);// c 是最终结果

            col2im_cpu(l.col_image, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta+i*l.c*l.h*l.w);
        }
    }
}

//当通过反向传播算法计算得到 bias和filter的更新量以后， 对他们进行更新
void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    // 这是对偏置进行更新
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);// update 偏置 learning_rate/batch是尺度
    scal_cpu(l.n, momentum, l.bias_updates, 1);//对偏置的增量进行尺度变换，这个冲量系数难道是为了衰减这个量对于下一个迭代的影响

    // 这是对filter进行更新，先计算filter的更新量，然后再更新filter
    axpy_cpu(size, -decay*batch, l.filters, 1, l.filter_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.filter_updates, 1, l.filters, 1);
    scal_cpu(size, momentum, l.filter_updates, 1);//对filter的增量进行尺度变换，这个冲量系数难道是为了衰减这个量对于下一个迭代的影响

}


//取出来第i个 filter
image get_convolutional_filter(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w,h,c,l.filters+i*h*w*c);
}

void rgbgr_filters(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_filter(l, i);
        if (im.c == 3) {
            rgbgr_image(im);// 将rgb的filter转换为bgr的filter
        }
    }
}

void rescale_filters(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_filter(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;//偏置在处理
        }
    }
}

//得到全部的filters
image *get_filters(convolutional_layer l)
{
    image *filters = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        filters[i] = copy_image(get_convolutional_filter(l, i));// 取出每一个filter，放到filters当中
        //normalize_image(filters[i]);
    }
    return filters;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_filters)
{
    image *single_filters = get_filters(l);
    show_images(single_filters, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_filters;
}

