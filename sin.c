#include <stdio.h>
#include <math.h>
#include <malloc.h>
#define N 10000000
#define M_PI 3.14159265358979323846


int main() {
	float* sinus = (float*)calloc(N, sizeof(float));
	for (size_t i = 0; i < N;++i)
		sinus[i] = sin((2 * M_PI) / N * i);
	float sum = 0;
#pragma copyin(sinus)
	{
#pragma acc kernels
		{

			for (size_t i = 0; i < N;++i)
				sum += sinus[i];
		}
	}
	printf("%e", sum);
	return 0;
}
