#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "activations.h"

struct layer;
typedef struct layer layer;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    CRNN
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, SMOOTH
} COST_TYPE;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    int batch_normalize;// 是否要进行Batch Normalize？
    int shortcut;
    int batch;// sample Num of this batch train
    int forced;
    int flipped;
    int inputs;// input 数据的维度 w＊h＊c
    int outputs;//output 的维度
    int truths;
    int h,w,c;//这应该是上一层的输出作为这里的输入，也就是特征图的大小w＊h，特征图的张数（或者叫通道数）c。
    int out_h, out_w, out_c;
    int n;//神经元的个数？ 其实这是输出的特征图的张数（或者叫通道数）out_c＝n， filter的个数就是c＊n个(或者将c个通道看做一个，那就是n个了)
    int groups;
    int size;//filter的大小，这个和w h什么关系（filter就是神经元吧）
    int side;
    int stride;//卷积的时候的步长
    int pad;//边框大小
    int sqrt;
    int flip;
    int index;
    int binary;
    int steps;
    int hidden;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;

    int dontload;
    int dontloadscales;//这是不要尺度变化的标志吧，和Batch Normalize有关系吧

    float temperature;
    float probability;
    float scale;

    int *indexes;
    float *rand;
    float *cost;// 经过一个网络以后的损失，一般来讲取第一个
    float *filters;// c＊n个 filter 的size＊size大小的参数
    char  *cfilters;
    float *filter_updates;
    float *state;//

    float *binary_filters;

    float *biases;// Neural Units的偏置，有n个
    float *bias_updates;

    float *scales;
    float *scale_updates;

    float *weights;// 目前发现是在全连接层当中要使用到的
    float *weight_updates;

    float *col_image;// 为了便于卷积，专门将数据转化为n＊out_h*out_w,并且对应于各个filter
    int   * input_layers;
    int   * input_sizes;
    float * delta;// 对该层网络的layer 求梯度得到的
    float * output;// 输出数据
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;// batch normalize mean
    float * variance;// batch normalize variance    (after train)

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;//可能也是和Batch Normalize又关系  (state.train=flase)
    float * rolling_variance;

    float * x;// 可做输出数据的暂存位置
    float * x_norm;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    #ifdef GPU
    int *indexes_gpu;
    float * state_gpu;
    float * filters_gpu;
    float * filter_updates_gpu;

    float *binary_filters_gpu;
    float *mean_filters_gpu;

    float * spatial_mean_gpu;
    float * spatial_variance_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * spatial_mean_delta_gpu;
    float * spatial_variance_delta_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * col_image_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;

    float * output_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
    #endif
};

void free_layer(layer);

#endif
