#include <iostream>
#include <stdlib.h>

using namespace std;

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
#pragma	acc data copy(arr) create(arr1, err)
	{
		while ((err > acc) && (iter < iters)) {
			iter++;		
			if ((iter % 100 == 0) || (iter == 1)){ // every 100 iterations we nullify error and compute it
            #pragma acc kernels async(1) // asynchronous computations on a new thread
            {
            err = 0;
            #pragma acc loop independent collapse(2) reduction(max:err) // collapsing double for into one
            for (int j = 1; j < n-1; j++)
                for (int i = 1; i < n-1; i++){
                    arr1[i][j] = arr[i][j] + 1./(double)(n*n*4) * ((arr[i - 1][j] - 2 * arr[i][j] + arr[i + 1][j]) + (arr[i][j - 1] - 2 * arr[i][j] + arr[i][j + 1])) / (h * h);
                    err = max(err, arr1[i][j] - arr[i][j]);
                }
            }
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
		if ((iter % 100 == 0) || (iter == 1)){ // every 100 iterations:
        #pragma acc wait(1) // synchronizing all threads
        #pragma acc update self(err) // updating error value on CPU
		}
	}
	}
	cout << iter << ' ' << err;
	return 0;
}
