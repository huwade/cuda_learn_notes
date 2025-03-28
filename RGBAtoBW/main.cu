#include <cuda_runtime.h>
#include "test_bmp.h"
#include <string>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define BLUR_SIZE 10

void cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

__global__ void rgbaToBwCUDA(uint32_t *data, size_t W, size_t H)
{
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_col_idx < W && C_row_idx < H)
    {
        size_t idx = C_row_idx * W + C_col_idx;
        // Cast the uint32_t pointer to a unsigned char pointer for easier manipulation
        unsigned char *pixel = (unsigned char *)(data + idx);

        // Extract RGB values
        unsigned char r = *(pixel + 2);
        unsigned char g = *(pixel + 1);
        unsigned char b = *(pixel);

        // Calculate the grayscale value using the given weights
        unsigned char bw = (unsigned char)(r * 0.299f + g * 0.587f + b * 0.114f);

        // Set the RGB values to the grayscale value
        *(pixel + 2) = bw;
        *(pixel + 1) = bw;
        *(pixel) = bw;
    }
}

__global__ void blurKernel(uint32_t *data, uint32_t *output, size_t W, size_t H)
{
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_col_idx < W && C_row_idx < H)
    {

        int pixels{0};
        int r_acc = 0, g_acc = 0, b_acc = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++)
        {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++)
            {
                int curRow = C_row_idx + blurRow;
                int curCol = C_col_idx + blurCol;
                if (curRow > -1 && curRow < H && curCol > -1 && curCol < W)
                {
                    ssize_t idx = curRow * W + curCol;
                    uint32_t pixel = data[idx];

                    // Extract RGB values
                    unsigned char r = (pixel >> 16) & 0xFF;
                    unsigned char g = (pixel >> 8) & 0xFF;
                    unsigned char b = pixel & 0xFF;

                    r_acc += r;
                    g_acc += g;
                    b_acc += b;
                    pixels++;
                }
            }
        }
        // Avoid division by zero
        if (pixels > 0)
        {
            r_acc /= pixels;
            g_acc /= pixels;
            b_acc /= pixels;
        }

        uint32_t newPixel = (r_acc << 16) | (g_acc << 8) | b_acc;
        size_t idx = C_row_idx * W + C_col_idx;
        output[C_row_idx * W + C_col_idx] = newPixel;
    }
}

int main()
{
    char openfile[] = "./pictures/01.bmp";
    char savefile[] = "./pictures/01_after.bmp";

    clock_t start = 0;
    clock_t end = 0;
    int loop = 1;
    BMP *bmp = (BMP *)malloc(sizeof(BMP));

    /* Load the image and print the infomation */
    bmpLoad(bmp, openfile);

    // std::cout << "RGBA to BW Start" << std::endl;
    // long stride = bmp->width * 4;
    // start = clock();
    // for (int i{0}; i < loop; i++)
    // {
    //     rgbaToBw(bmp, bmp->width, bmp->height, stride);
    // }
    // end = clock();

    // std::cout << "Execution time of rgbaToBw(): "
    //           << static_cast<double>(end - start) / CLOCKS_PER_SEC / loop
    //           << " seconds" << std::endl;

    uint32_t *d_input{nullptr};
    uint32_t *d_output{nullptr};
    cudaCheck(cudaMalloc((void **)&d_input, sizeof(uint32_t) * bmp->width * bmp->height));
    // for average blur
    cudaCheck(cudaMalloc((void **)&d_output, sizeof(uint32_t) * bmp->width * bmp->height));
    cudaCheck(cudaMemcpy(d_input, bmp->data, sizeof(uint32_t) * bmp->width * bmp->height, cudaMemcpyHostToDevice));
    // image is 1920*1080, block dim is 32*32,
    // This results in a grid dimension of (1920 + 31) / 32 = 60 blocks in the x-dimension and (1080 + 31) / 32 = 34 blocks in the y-dimension.
    // Thus, your grid dimension would be grid_dim{60, 34, 1}.
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(bmp->width) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(bmp->height) + block_dim.y - 1U) / block_dim.y, 1U};

    start = 0;
    end = 0;

    std::cout << "Cuda Start" << std::endl;
    start = clock();
    for (int i{0}; i < loop; i++)
    {
        // rgbaToBwCUDA<<<grid_dim, block_dim>>>(d_input, bmp->width, bmp->height);
        blurKernel<<<grid_dim, block_dim>>>(d_input, d_output, bmp->width, bmp->height);
    }
    end = clock();
    // Synchronize and check for errors
    cudaDeviceSynchronize();

    cudaMemcpy(bmp->data, d_output, sizeof(uint32_t) * bmp->width * bmp->height, cudaMemcpyDeviceToHost);
    bmpSave(bmp, savefile);
    cudaFree(d_output);
    cudaFree(d_input);
    std::cout << "Execution time of rgbaToBwCUDA(): "
              << static_cast<double>(end - start) / CLOCKS_PER_SEC / loop
              << " seconds" << std::endl;
}