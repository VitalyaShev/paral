#include <iostream>
#include <stdlib.h>
#include <cub/cub.cuh>


using namespace std;

__global__ void compute(double* Carr1, double* Carr, int n, double h){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if((i > 0) && (i < n - 1) && (j > 0) && (j < n - 1))
        Carr1[i*n+j] = Carr[i*n+j] + 1./(double)(n*n*4) * ((Carr[(i - 1)*n+j] - 2 * Carr[i*n+j] + Carr[(i + 1)*n+j]) + (Carr[i*n + j - 1] - 2 * Carr[i*n+j] + Carr[i*n +j + 1])) / (h * h);
}

__global__ void Max_Reduction(double* Carr, double* Carr1, int n, double* BlockErr){
    typedef cub::BlockReduce<double, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 16> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double thread_data=0.0;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
    if ((i > 0) && (i < n - 1) && (j > 0) && (j < n - 1))
        thread_data = Carr1[i*n + j] - Carr[i*n + j];
    double aggregate = BlockReduce(temp_storage).Reduce(thread_data, cub::Max());
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        BlockErr[blockIdx.y*gridDim.x + blockIdx.x] = aggregate;
}


int main() {
	int iter;
	double acc = 0.000001; 
	double err = 1;
	int n = 1024; 
	int iters = 1000000; 
	double step = 10. / (double)(n);
	iter = 0;
	double arr[n][n], arr1[n][n];
	double h = 1.0 / (double)n;
	arr[0][0] = arr1[0][0] = 30;
	arr[n-1][n - 1] = arr1[n - 1][n - 1] = 20;
	arr[0][n - 1] = arr1[0][n - 1] = 20;
	arr[n - 1][0] = arr1[n - 1][0] = 10;
	for (int i = 1; i < n - 1; i++) {
		arr[0][i] = arr[0][i - 1] + step;
		arr[n - 1][i] = arr[n - 1][i - 1] + step;
	}
	for (int j = 1; j < n - 1; j++) {
		arr[j][0] = arr[j - 1][0] + step;
		arr[j][n - 1] = arr[j - 1][n - 1] + step;
	}
	for (int i = 1; i < n - 1; i++)
		for (int j = 1; j < n - 1; j++)
			arr[i][j] = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
            arr1[i][j] = arr[i][j];
    double* Carr;
    double* Carr1;
    cudaMalloc(&Carr, n*n*sizeof(double));
	cudaMalloc(&Carr1, n*n*sizeof(double));

    dim3 BS = dim3(16,16);
	dim3 GS = dim3(ceil(n/(float)BS.x),ceil(n/(float)BS.y));

    double* CBlockErr;
    cudaMalloc(&CBlockErr, GS.x*GS.y*sizeof(double));
    double BlockErr[GS.x*GS.y];

    cudaMemcpy(Carr, arr, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Carr1, arr, n*n*sizeof(double), cudaMemcpyHostToDevice);
		while ((err > acc) && (iter < iters)) {
			iter+=2;
            compute<<<GS,BS>>>(Carr1, Carr, n, h);
            compute<<<GS,BS>>>(Carr, Carr1, n, h);		
			if ((iter % 100 == 0) || (iter == 2)){  
            err = 0;
        	Max_Reduction<<<GS,BS>>>(Carr1, Carr, n, CBlockErr);
            cudaDeviceSynchronize();
            cudaMemcpy(BlockErr, CBlockErr, GS.x*GS.y*sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < GS.x; i++)
                for (int j = 0; j < GS.y; j++)
                    err = max(err, BlockErr[i*GS.x + j]);
            }
        }        
	cout << iter << ' ' << err;
    cudaFree(Carr);
	cudaFree(Carr1);
    cudaFree(CBlockErr);
    cudaDeviceReset();
	return 0;
}
