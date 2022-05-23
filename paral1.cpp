#include <iostream>
#include <stdlib.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

using namespace std;

int main() {
	int iter;
	double acc = 0.000001; 
	double err = 1;
	int n = 256; 
	int iters = 1000000; 
	double step = 10. / (double)(n);
	iter = 0;
	double arr[n][n], arr1[n][n];
	double h = 1.0 / (double)n;
	arr[0][0] = arr1[0][0] = 30;
	arr[n-1][n - 1] = arr1[n - 1][n - 1] = 20;
	arr[0][n - 1] = arr1[0][n - 1] = 20;
	arr[n - 1][0] = arr1[n - 1][0] = 10;
	#pragma acc parallel
	{
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
	}
	cublasHandle_t handle;
    cublasCreate(&handle);
    double temp[n*n];
#pragma	acc data copy(arr) create(arr1, err, temp)
	{
        #pragma acc kernels
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
            arr1[i][j] = arr[i][j];
		while ((err > acc) && (iter < iters)) {
			iter++;		
			if ((iter % 100 == 0) || (iter == 1)){ // every 100 iterations we nullify error and compute it
            #pragma acc kernels async(1) // asynchronous computations on a new thread
            {
            err = 0;
            for (int j = 1; j < n-1; j++)
                for (int i = 1; i < n-1; i++)
                    arr1[i][j] = arr[i][j] + 1./(double)(n*n*4) * ((arr[i - 1][j] - 2 * arr[i][j] + arr[i + 1][j]) + (arr[i][j - 1] - 2 * arr[i][j] + arr[i][j + 1])) / (h * h);
            }
			#pragma acc wait(1)
            int idx;
            //making addresses of device data avaiable on the host
            #pragma acc host_data use_device(arr1, arr, temp)
            {
            // here we will be implementing the following expression:
            //  err = max(err, Anew[i][j] - A[i][j]);
            double alpha = -1.;
            for (int i = 0; i < n; i++){
                // copying Anew matrix to a cublas-like array temp
                cublasDcopy(handle, n, arr1[i], 1, &temp[i*n], 1);
                // multiplying A matrix with -1 and adding it to temp
                cublasDaxpy(handle, n, &alpha, arr[i], 1, &temp[i*n], 1);
            }
            
            // searching for a max error
            cublasIdamax(handle, n*n, temp, 1, &idx);
            }
            
            // updating temp on CPU
            #pragma acc update self(temp[idx-1:1])
            
            // getting the result
            err = fabs(temp[idx-1]);
        } 
		else{
            #pragma acc kernels async(1)
            #pragma acc loop independent collapse(2)
            for (int j = 1; j < n-1; j++)
                for (int i = 1; i < n-1; i++){
                    arr1[i][j] = arr[i][j] + 1./(double)(n*n*4) * ((arr[i - 1][j] - 2 * arr[i][j] + arr[i + 1][j]) + (arr[i][j - 1] - 2 * arr[i][j] + arr[i][j + 1])) / (h * h);
                }
        }
		#pragma acc kernels async(1) // updating matrix
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                arr[i][j] = arr1[i][j];
		}
	}
	cout << iter << ' ' << err;
    cublasDestroy(handle);
	return 0;
}
