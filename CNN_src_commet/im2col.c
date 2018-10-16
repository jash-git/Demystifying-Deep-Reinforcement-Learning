#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;//在边界上的像素点 返回0
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE

// input data_im, output data_col 实现将image转化为 便于 卷积计算的数据结构
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;// ksize 是什么？
    int width_col = (width - ksize) / stride + 1;
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    //将数据按照channels_col个通道去存储，每一个通道的大小＝output中一个feature Map的大小
    //这样做的好处是可以便于卷积操作, 因为现在的每一个通道的value都是filter当中的一个值 要在这个位置去计算的
    int channels_col = channels * ksize * ksize;

    //经过三个循坏以后生成的是一个channels_col＊ output_h* output_w 大小的数据，卷积非常的方便（实际上输入的大小只有height＊width）
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;//相当于(w_offset, h_offset)这个位置的filter point
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride; //要取的值的位置，其实就是这个filter的值要卷积的地方
                int im_col = w_offset + w * stride;//遍历到这个point要相乘的 
                int col_index = (c * height_col + h) * width_col + w; // 取出来的值放的位置，这个很明显了。
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

