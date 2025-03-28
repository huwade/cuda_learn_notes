#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

// error checking macro
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

struct list_elem
{
    int key;
    list_elem *next;
};

template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes)
{
    cudaMallocManaged(&ptr, num_bytes);
}

__host__ __device__ void print_element(list_elem *list)
{
    list_elem *head = list;
    while (head)
    {
        printf("key = %d\n", head->key);
        head = head->next;
    }
}

__global__ void gpu_print_element(list_elem *list)
{
    print_element(list);
}

const int num_elem = 5;

int main()
{

    list_elem *list_base, *list;
    alloc_bytes(list_base, sizeof(list_elem));
    list = list_base;

    for (int i = 0; i < num_elem; i++)
    {
        list->key = i;
        alloc_bytes(list->next, sizeof(list_elem));
        list = list->next;
    }

    print_element(list_base);
    gpu_print_element<<<1, 1>>>(list_base);
    cudaDeviceSynchronize();
    cudaCheckErrors("cuda error!");
}
