//Do NOT MODIFY THIS FILE

// #include "knn.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
// #include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/extrema.h>

#include "gputimer.h"
#include "gpuerrors.h"

#define D 128
#define D_L 100
#define N_ref 1000000

// ===========================> Functions Prototype <===============================
int fvecs_read (const char *fname, int d, int n, float *a);
int ivecs_read (const char *fname, int d, int n, int *a);
int ivecs_write (const char *fname, int d, int n, const int *v);
int calc_error(int* nn_true, int* nn_predict, unsigned int N, unsigned int K);
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K);
void gpuKernels(float* ref, float* query, int* pred, unsigned int N, unsigned int K, double* gpu_kernel_time, int* sort_val);
// void gpuKernels(float* ref, float* query, int* pred, float* a, unsigned int N, unsigned int K, double* gpu_kernel_time);


// =================================================================================
__global__ void kernelFunc1(float* d_ref, float* d_q , float* d_dist) {
    __shared__ float sum_arr[128];
    int block, index;
      long long i;
    i =  blockIdx.y * gridDim.x + blockIdx.x;   //y * 1000 + x
    block = i;
    i *= blockDim.x * blockDim.y * blockDim.z;  //128 * 1 * 1
    i += threadIdx.x;
    index = threadIdx.x;
  
    sum_arr[index] = pow(*(d_ref + i) - *(d_q + index), 2);
  
    __syncthreads(); 
  
    int counter = 128;
    while(counter >= 2){
      if(index < counter/2){
        sum_arr[index] += sum_arr[index + counter/2];
        __syncthreads();
      }
      counter = counter/2;    
    }
    d_dist[block] = sqrt(sum_arr[0]);
  }
  // =================================================================================
  __global__ void kernelFunc2(float* d_ref, float* d_q , float* d_dist) {
    __shared__ float sum_arr[128];
    int block, index;
      long long i;
    i =  blockIdx.y * gridDim.x + blockIdx.x;   //y * 1000 + x
    block = i;
    i *= blockDim.x * blockDim.y * blockDim.z;  //128 * 1 * 1
    i += threadIdx.x;
    index = threadIdx.x;
  
    sum_arr[index] = abs(*(d_ref + i) - *(d_q + index));
  
    __syncthreads(); 
  
    int counter = 128;
    while(counter >= 2){
      if(index < counter/2){
        sum_arr[index] += sum_arr[index + counter/2];
        __syncthreads();
      }
      counter = counter/2;    
    }
    // sum[block] = thrust::reduce(sum_arr, sum_arr + 128, 0, thrust::plus<int>());
    d_dist[block] = (sum_arr[0]);
  }
  // =================================================================================
__global__ void kernelFunc3(float* d_ref, float* d_q , float* d_dist) {
  __shared__ float mult[128];
  __shared__ float a_square[128];
  __shared__ float b_square[128];

  int block, index;
	long long i;
  i =  blockIdx.y * gridDim.x + blockIdx.x;   //y * 1000 + x
  block = i;
  i *= blockDim.x * blockDim.y * blockDim.z;  //128 * 1 * 1
  i += threadIdx.x;
  index = threadIdx.x;

  a_square[index] = pow(*(d_ref + i), 2);
  b_square[index] = pow(*(d_q + index), 2);
  mult[index] = (*(d_ref + i)) * (*(d_q + index));

  __syncthreads(); 

  int counter = 128;
  while(counter >= 2){
    if(index < counter/2){
      a_square[index] += a_square[index + counter/2];
      b_square[index] += b_square[index + counter/2];
      mult[index] += mult[index + counter/2];
      __syncthreads();
    }
    counter = counter/2;    
  }
  d_dist[block] = mult[0]/(sqrt(a_square[0] * b_square[0]));
}
// =================================================================================
__global__ void gpu_value_init(int* array){
  int block;
	long long i;
  i =  blockIdx.y * gridDim.x + blockIdx.x;   //y * 1000 + x
  block = i;
  array[block] = block;
}
// =================================================================================
int main(int argc, char *argv[]) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);

    // get parameters from command line
    unsigned int N, K;
    get_inputs(argc, argv, N, K);

    // allocate memory in CPU for calculation
    float* reference; // reference vectors
    float* query; // query points
    int* nn_true; // groundtruth for the query points
    int* nn_predict;
    // float* distances;
    int* sort_value;
    int index;

    // Memory Allocation
    reference  = (float*)malloc(N_ref * D * sizeof(float));
    query      = (float*)malloc(N * D * sizeof(float));
    nn_true    = (int*)malloc(D_L * N * sizeof(int));
    nn_predict = (int*)malloc(3 * K * N * sizeof(int));
    // distances  = (float*)malloc(N_ref * sizeof(float));
    sort_value = (int*)malloc(N_ref * sizeof(int));
    // fill references, query and labels with the values read from files
    fvecs_read("/home/data/ref.fvecs", D, N_ref, reference);
    fvecs_read("/home/data/query.fvecs", D, N, query);
    ivecs_read("/home/data/groundtruth.ivecs", D_L, N, nn_true);
    printf("ground truths read\n");

    for(int j=0; j<N_ref; j++){
      sort_value[j] = j;
    }
    // time measurement for GPU calculation
    double gpu_kernel_time = 0.0;
    clock_t t0 = clock();
    gpuKernels(reference, query, nn_predict, N, K, &gpu_kernel_time, sort_value);
      /* for(int i=0; i< 3 * N * K; i++){
      index = ((i/K) * D_L) + (i % K);
        if(i < N * K){
            printf("Euclidean\t N is %d, gth is: %d \t and prediction is: %d\n", i % K, nn_true[index], nn_predict[i]);
            if ( i % K == 0 ){
                printf("\n\nQuery num = %d", (i/K) % 3);
                printf("\n\n\n");
              }
        }
        else if(i < 2 * N * K){
            printf("Manhattan\t N is %d, gth is: NULL \t and prediction is: %d\n", i % K, nn_predict[i]);
            if ( i % K == 0 ){
                printf("\n\nQuery num = %d", (i/K) % 3);
                printf("\n\n\n");
              }
        }
        else{
            printf("Cosine\t N is %d, gth is: NULL \t and prediction is: %d\n", i % K, nn_predict[i]);
            if ( i % K == 0 ){
                printf("\n\nQuery num = %d", (i/K) % 3);
                printf("\n\n\n");
              }
        }
    } */
    
    clock_t t1 = clock();

    // check correctness of calculation
    double acc = 1 - (double)calc_error(nn_true, nn_predict, N, K);
    printf("k=%d n=%d GPU=%g ms GPU-Kernels=%g ms accuracy=%f\n",
    K, N, (t1-t0)/1000.0, gpu_kernel_time, acc);

    // write the output to a file
    ivecs_write("output.ivecs", K, N, nn_predict);
    
    // free allocated memory for later use
    free(reference);
    free(nn_predict);
    free(nn_true);
    free(query);

    return 0;
}

//-----------------------------------------------------------------------------
void gpuKernels(float* reference, float* query, int* nn_predict, unsigned int N, unsigned int K, double* gpu_kernel_time, int* sort_val) {
// void gpuKernels(float* reference, float* query, int* nn_predict, float* d, unsigned int N, unsigned int K, double* gpu_kernel_time) {

    // Memory Allocation and Copy to Device


	  GpuTimer timer;
    timer.Start();
    // printf("\nHello1");
    float* d_reference;
	  float* d_query;
    // int*   d_nn_predict;
    float* d_distances;
    // int* value;
    int* d_value;
    int* d_sort_val;
	
    HANDLE_ERROR(cudaMalloc((void**)&d_reference, N_ref * D * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_query, N * D * sizeof(float)));
    // HANDLE_ERROR(cudaMalloc((void**)&d_nn_predict, 3 * D_L * N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_distances, N_ref * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value, N_ref * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_sort_val, N_ref * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(d_reference, reference, N_ref * D * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_query, query, N * D * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_sort_val, sort_val, N_ref * sizeof(int), cudaMemcpyHostToDevice));

    // dim3 grid ( 512 * 128, 1, 1 ); // This is not correct because 512 * 128 65536 > 65535
    dim3 grid (1000, 1000, 1); //x,y,z
    dim3 block (128, 1, 1); //x,y,z

    for(int m=0; m<N; m++){
      HANDLE_ERROR(cudaMemcpy(d_sort_val, sort_val, N_ref * sizeof(int), cudaMemcpyHostToDevice));
      kernelFunc1<<< grid , block>>>(d_reference, &(d_query[m * 128]), d_distances);
      // gpu_value_init<<< grid, block>>>(d_sort_val);
      thrust::device_ptr<float> t_dist(d_distances);
      thrust::device_ptr<int> t_value(d_sort_val);
      thrust::sort_by_key(t_dist, t_dist + N_ref, t_value);
      HANDLE_ERROR(cudaMemcpy(&(nn_predict[m * K]), d_sort_val, K * sizeof(int), cudaMemcpyDeviceToHost));
    }

    for(int m=0; m<N; m++){
      HANDLE_ERROR(cudaMemcpy(d_sort_val, sort_val, N_ref * sizeof(int), cudaMemcpyHostToDevice));
      // gpu_value_init<<< grid, block>>>(d_sort_val);
      kernelFunc2<<< grid , block>>>(d_reference, &(d_query[m * 128]), d_distances);
      
      thrust::device_ptr<float> t_dist(d_distances);
      thrust::device_ptr<int> t_value(d_sort_val);

      thrust::sort_by_key(t_dist, t_dist + N_ref, t_value);
      HANDLE_ERROR(cudaMemcpy(&(nn_predict[(m + N) * K]), d_sort_val, K * sizeof(int), cudaMemcpyDeviceToHost));  
        
      }

    for(int m=0; m<N; m++){
      HANDLE_ERROR(cudaMemcpy(d_sort_val, sort_val, N_ref * sizeof(int), cudaMemcpyHostToDevice));
      kernelFunc3<<< grid , block>>>(d_reference, &(d_query[m * 128]), d_distances);
      // gpu_value_init<<< grid, block>>>(d_sort_val);
      thrust::device_ptr<float> t_dist(d_distances);
      thrust::device_ptr<int> t_value(d_sort_val);

      thrust::sort_by_key(t_dist, t_dist + N_ref, t_value);
      HANDLE_ERROR(cudaMemcpy(&(nn_predict[(m + 2*N) * K]), d_sort_val, K * sizeof(int), cudaMemcpyDeviceToHost));
      // HANDLE_ERROR(cudaMemcpy(&(nn_predict[(m + 2*N) * K]), &(d_sort_val[N_ref - K - 1]), K * sizeof(int), cudaMemcpyDeviceToHost));

    }
    
    cudaFree(d_reference);
    cudaFree(d_query);
    // cudaFree(d_nn_predict);
    cudaFree(d_distances);
    cudaFree(d_value);
    cudaFree(d_sort_val);
  	timer.Stop();
	  *gpu_kernel_time = timer.Elapsed();

    //Copy to Host and Free the Memory

}
//-----------------------------------------------------------------------------
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K)
{
    if (
	argc != 3 ||
	atoi(argv[1]) < 0 || atoi(argv[1]) > 10000 ||
	atoi(argv[2]) < 0 || atoi(argv[2]) > 100
	) {
        printf("<< Error >>\n");
        printf("Enter the following command:\n");
        printf("\t./knn  N  K\n");
        printf("\t\tN must be between 0 and 10000\n");
        printf("\t\tK must be between 0 and 100\n");
		exit(-1);
    }
	N = atoi(argv[1]);
	K = atoi(argv[2]);
}
//-----------------------------------------------------------------------------
int fvecs_read (const char *fname, int d, int n, float *a)
{
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "fvecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  long i;
  for (i = 0; i < n; i++) {
    int new_d;

    if (fread (&new_d, sizeof (int), 1, f) != 1) {
      if (feof (f))
        break;
      else {
        perror ("fvecs_read error 1");
        fclose(f);
        return -1;
      }
    }

    if (new_d != d) {
      fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
      fclose(f);
      return -1;
    }

    if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
      fprintf (stderr, "fvecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return i;
}

int ivecs_read (const char *fname, int d, int n, int *a)
{
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "ivecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  long i;
  for (i = 0; i < n; i++) {
    int new_d;

    if (fread (&new_d, sizeof (int), 1, f) != 1) {
      if (feof (f))
        break;
      else {
        perror ("ivecs_read error 1");
        fclose(f);
        return -1;
      }
    }

    if (new_d != d) {
      fprintf (stderr, "ivecs_read error 2: unexpected vector dimension\n");
      fclose(f);
      return -1;
    }

    if (fread (a + d * (long) i, sizeof (int), d, f) != d) {
      fprintf (stderr, "ivecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return i;
}

int ivecs_write (const char *fname, int d, int n, const int *v)
{
  FILE *f = fopen (fname, "w");
  if (!f) {
    perror ("ivecs_write");
    return -1;
  }

  int i;
  for (i = 0 ; i < n ; i++) {
    fwrite (&d, sizeof (d), 1, f);
    fwrite (v, sizeof (*v), d, f);
    v+=d;
  }
  fclose (f);
  return n;
}

int calc_error(int* nn_true, int* nn_predict, unsigned int N, unsigned int K) {
    // int fault = 0;
    int sum = 0;
    int index;
    for(int n=0; n< N * K; n++){
      index = ((n/K) * D_L) + (n % K);
      // printf("\n%d\n", index);
        if(nn_predict[n] != nn_true[index]){
            sum += 1;
        }
    }
    return sum/(N * K);
}