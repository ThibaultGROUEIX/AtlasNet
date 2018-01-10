#include <THC/THC.h>
#include "nnd_cuda.h"



extern THCState *state;


int nnd_forward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *dist1, THCudaTensor *dist2, THCudaIntTensor *idx1, THCudaIntTensor *idx2) {
    int success = 0;
    success = NmDistanceKernelLauncher(xyz1->size[0],
	xyz1->size[1],
	THCudaTensor_data(state, xyz1),
	xyz2->size[1],
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, dist1),
	THCudaIntTensor_data(state, idx1),
	THCudaTensor_data(state, dist2),
	THCudaIntTensor_data(state, idx2),
	THCState_getCurrentStream(state)
	);
	//int NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream)
		
    
    if (!success) {
    THError("aborting");
    }
    return 1;
}


int nnd_backward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *gradxyz1, THCudaTensor *gradxyz2, THCudaTensor *graddist1, 
					  THCudaTensor *graddist2, THCudaIntTensor *idx1, THCudaIntTensor *idx2) {
    
    int success = 0;
    success = NmDistanceGradKernelLauncher(xyz1->size[0],
	xyz1->size[1],
	THCudaTensor_data(state, xyz1),
	xyz2->size[1],
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, graddist1),
	THCudaIntTensor_data(state, idx1),
	THCudaTensor_data(state, graddist2),
	THCudaIntTensor_data(state, idx2),
	THCudaTensor_data(state, gradxyz1),
	THCudaTensor_data(state, gradxyz2),
	THCState_getCurrentStream(state)	
	);
	//int NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream)

    if (!success) {
    THError("aborting");
    }

    return 1;
}



