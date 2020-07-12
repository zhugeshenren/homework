/*
完整的向量点积CUDA程序
a=[a1,a2,…an], b=[b1,b2,…bn]
a*b=a1*b1+a2*b2+…+an*bn
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <time.h>
#include <cuda.h> 
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h> 
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <numeric>
#include <algorithm>

using namespace std;

#define COLS 9298
#define ROWS 256

__global__ void cuda_fun(double *vec, 
    size_t *indexs, 
    size_t offset, 
    double * ptr_th_count,
    double * ptr_th_sum,
    size_t * ptr_th_count_idx,
    double * ptr_th_count_value
    ) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("%d ,%d, %d, %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    // printf("%d\n", idx);

    size_t begin = idx * offset;
    size_t end = idx * offset + offset;
    
    double count = 0;
    double value = vec[begin];
    size_t i = begin;

    for (; i < end; i++)
    {
        if (vec[i] == value)
        {
            count++;
        }
        else
        {
            value = vec[i];
            break;
        }
    }

    // printf("count1 : %f\n", count);
    //printf("%d\n", idx);
    ptr_th_count[idx * 2] = count;
    ptr_th_count_idx[idx * 2] = indexs[i-1] / COLS;
    ptr_th_count_value[idx * 2] = vec[i-1];
    
    count = 0;
    for (; i < end; i++)
    {
        if (value == vec[i]) {
            count++;
        }
        else
        {
            value = vec[i];
            size_t index = indexs[i];
            
            index /= COLS;
            ptr_th_sum[index] += -(count / COLS) * log2(count / COLS);
            count = 1;
        }
    }

    ptr_th_count[idx * 2 + 1] = count;

    //printf("count2 : %f\n", count);

    ptr_th_count_idx[idx * 2 + 1] = indexs[i - 1] / COLS;
    ptr_th_count_value[idx * 2 + 1] = vec[i - 1];
};

__global__ void merge(double* ptr_th_count, double* ptr_th_sum, size_t* ptr_th_count_idx, double *ptr_th_count_value) {
    int length = 4649 * 256 * 2;
    double count = 0.0;
    double value = ptr_th_count_value[0];
    size_t index = ptr_th_count_idx[0];

    for (size_t i = 0; i < length; i++)
    {
        if (ptr_th_count_value[i] == value && index == ptr_th_count_idx[i]) {
            count += ptr_th_count[i];
            printf("count %f\n", count);
        }
        else
        {
            ptr_th_sum[index] += - (count / COLS) * log2(count / COLS);
            count = ptr_th_count[i];
            index = ptr_th_count_idx[i];
            value = ptr_th_count_value[i];
        }
    }
    //ptr_th_sum[index] += -(count / COLS) * log2(count / COLS);

    for (size_t i = 0; i < 256; i++)
    {
        printf("%d , %f,\n", i, ptr_th_sum[i]);
    }
    printf("\n");
}

void load_data(size_t& rows, size_t& cols, double*& x, double*& y) {
    string x_file_path = "D:/FCBF_DATA/x.bin";
    string y_file_path = "D:/FCBF_DATA/y.bin";

    cols = 256;
    rows = 9298;

    size_t x_length = (size_t)cols * rows;

    double* read_x = (double*)malloc(x_length * sizeof(double));

    ifstream in_x(x_file_path, ios::in | ios::binary);

    in_x.read((char*)(read_x), (sizeof(double)) * x_length);

    x = (double*)malloc(x_length * sizeof(double));

    for (int i = 0; i < cols; i++) {
        for (size_t j = 0; j < rows; j++) {
            x[j + i * rows] = read_x[j * cols + i];
        }
    }

    in_x.close();

    y = (double*)malloc(sizeof(double) * rows);

    ifstream in_y(y_file_path, ios::in | ios::binary);
    in_y.read((char*)y, rows * (sizeof(double)));
    in_y.close();

    // 扩充数据集
    size_t mul = 1;
    double* new_x = (double*)malloc(sizeof(double) * rows * cols * mul);
    for (size_t i = 0; i < mul; i++)
    {
        for (size_t j = 0; j < x_length; j++)
        {
            new_x[x_length * i + j] = x[j];
        }
    }

    rows = 256*mul;
    cols = 9298;
    free(x);
    free(read_x);
    x = new_x;
}


void parallel_merge() {

}

void entropy(size_t& rows, size_t& cols, double* x, double* y) {
    size_t* index_x;
    size_t* index_y;
    clock_t end, start;

    index_x = (size_t*)malloc(sizeof(size_t) * rows * cols);
    index_y = (size_t*)malloc(sizeof(size_t) * cols);

    thrust::device_vector<size_t> d_index_x(cols * rows);
    thrust::device_vector<double> d_x(x, x + cols * rows);

    thrust::device_vector<size_t> d_index_y(cols);
    thrust::device_vector<double> d_y(y, y + cols);

    
    thrust::sequence(d_index_x.begin(), d_index_x.end());
    thrust::sequence(d_index_y.begin(), d_index_y.end());

    start = clock();
    thrust::sort_by_key(d_x.begin(), d_x.end(), d_index_x.begin());
    thrust::sort_by_key(d_y.begin(), d_y.end(), d_index_y.begin());

    end = clock();
    printf("sort : %d\n", end - start);

    double* ptr_x = thrust::raw_pointer_cast(d_x.data());
    size_t* ptr_index_x = thrust::raw_pointer_cast(d_index_x.data());

    double* ptr_y = thrust::raw_pointer_cast(d_y.data());
    size_t* ptr_index_y = thrust::raw_pointer_cast(d_index_y.data());

    thrust::device_vector<double>th_count(4649 * 256 * 2);
    thrust::fill(th_count.begin(), th_count.end(), 0);
    double* ptr_th_count = thrust::raw_pointer_cast(th_count.data());

    thrust::device_vector<double>th_sum(rows);
    thrust::fill(th_sum.begin(), th_sum.end(), 0);
    double* ptr_th_sum = thrust::raw_pointer_cast(th_sum.data());

    thrust::device_vector<size_t> th_count_idx(4649 * 256 * 2);
    size_t* ptr_th_count_idx = thrust::raw_pointer_cast(th_count_idx.data());

    thrust::device_vector<double> th_count_value(4649 * 256 * 2);
    double* ptr_th_count_value = thrust::raw_pointer_cast(th_count_value.data());

    size_t offset = 2;

    start = clock();

    // 464,9 100 256
    cuda_fun << <4649, 256 >> > (ptr_x,
        ptr_index_x, 
        offset, 
        ptr_th_count,
        ptr_th_sum,
        ptr_th_count_idx,
        ptr_th_count_value);

    cudaDeviceSynchronize();

    merge << <1, 1 >> > (ptr_th_count, ptr_th_sum, ptr_th_count_idx, ptr_th_count_value);
    cudaDeviceSynchronize();
    end = clock();

    printf("cuda time : %d \n", end - start);

   
};

int main()
{
    clock_t end, start;

    cudaError_t res;
    size_t rows, cols;

    double* x;
    size_t* index_x;

    double* y;
    size_t* index_y;
    

    load_data(rows, cols, x, y);

    start = clock();
    entropy(rows, cols, x, y);
    end = clock();
    printf("entropy time %d \n", end - start);

    start = clock();
    sort(x, x+cols*rows);
    end = clock();

    printf("stl sort: %d \n", end - start);

    return 0;
}