int nnd_forward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *dist1, THCudaTensor *dist2, THCudaIntTensor *idx1, THCudaIntTensor *idx2);


int nnd_backward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *gradxyz1, THCudaTensor *gradxyz2, THCudaTensor *graddist1, THCudaTensor *graddist2, THCudaIntTensor *idx1, THCudaIntTensor *idx2);

