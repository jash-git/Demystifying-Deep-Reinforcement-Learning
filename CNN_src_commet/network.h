// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "layer.h"
#include "data.h"

//在一个网络当中，network_state 包括了一个network，训练数据input，数据的真实值truth（用于计算损失）等。
// network结构体包括了 各层神经网络，样本信息，迭代信息等等

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG
} learning_rate_policy;

typedef struct network{
    int n;//这是network的层数
    int batch;//样本个数
    int *seen;// 当前训练过的样本的总数  每train一次就增加一个batch
    float epoch;//迭代次数
    int subdivisions;// 确定多少批（batch）的时候更新
    float momentum;// 冲量，根据它来保留上一次迭代的梯度，继续对下一次影响。
    float decay;// 延迟量， 降低改次剃度的影响
    layer *layers;//其中所包含的layers
    int outputs;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;

    int inputs;// 输入数据相关
    int h, w, c;
    int max_crop;

    #ifdef GPU
    float **input_gpu;
    float **truth_gpu;
    #endif
} network;

//state  gets net 
typedef struct network_state {
    float *truth;// 数据 的 标签，或者是真实值
    float *input;// 数据的预测值  （每一层的输入都存起来了）
    float *delta;// 数据的 变化量
    int train; // 标志位
    int index;// layer的编号
    network net;// 网络
} network_state;

#ifdef GPU
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);
void forward_network_gpu(network net, network_state state);
void backward_network_gpu(network net, network_state state);
void update_network_gpu(network net);
#endif

float get_current_rate(network net);
int get_current_batch(network net);
void free_network(network net);
void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net, network_state state);
void backward_network(network net, network_state state);
void update_network(network net);

float train_network(network net, data d);
float train_network_batch(network net, data d, int n);
float train_network_sgd(network net, data d, int n);
float train_network_datum(network net, float *x, float *y);

matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);

int get_network_nuisance(network net);
int get_network_background(network net);

#endif

