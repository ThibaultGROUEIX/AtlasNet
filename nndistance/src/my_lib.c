#include <TH/TH.h>


void nnsearch(int b,int n,int m,const float * xyz1,const float * xyz2,float * dist,int * idx){
    for (int i=0;i<b;i++){
        for (int j=0;j<n;j++){
            float x1=xyz1[(i*n+j)*3+0];
            float y1=xyz1[(i*n+j)*3+1];
            float z1=xyz1[(i*n+j)*3+2];
            double best=0;
            int besti=0;
            for (int k=0;k<m;k++){
                float x2=xyz2[(i*m+k)*3+0]-x1;
                float y2=xyz2[(i*m+k)*3+1]-y1;
                float z2=xyz2[(i*m+k)*3+2]-z1;
                double d=x2*x2+y2*y2+z2*z2;
                if (k==0 || d<best){
                    best=d;
                    besti=k;
                }
            }
            dist[i*n+j]=best;
            idx[i*n+j]=besti;
        }
    }
}

int nnd_forward(THFloatTensor *xyz1, THFloatTensor *xyz2, THFloatTensor *dist1, THFloatTensor *dist2, THIntTensor *idx1, THIntTensor *idx2) {
    int batchsize = xyz1->size[0];
    int n = xyz1->size[1];
    int m = xyz2->size[1];
    
    //printf("%d %d %d\n", batchsize, n, m);
    
    float *xyz1_data = THFloatTensor_data(xyz1);
    float *xyz2_data = THFloatTensor_data(xyz2);
    float *dist1_data = THFloatTensor_data(dist1);
    float *dist2_data = THFloatTensor_data(dist2);
    int *idx1_data = THIntTensor_data(idx1);
    int *idx2_data = THIntTensor_data(idx2);
     
    nnsearch(batchsize, n, m, xyz1_data, xyz2_data, dist1_data, idx1_data);
    nnsearch(batchsize, m, n, xyz2_data, xyz1_data, dist2_data, idx2_data);
    
    return 1;
}


int nnd_backward(THFloatTensor *xyz1, THFloatTensor *xyz2, THFloatTensor *gradxyz1, THFloatTensor *gradxyz2, THFloatTensor *graddist1, THFloatTensor *graddist2, THIntTensor *idx1, THIntTensor *idx2) {
    
    int b = xyz1->size[0];
    int n = xyz1->size[1];
    int m = xyz2->size[1];
    
    //printf("%d %d %d\n", batchsize, n, m);
    
    float *xyz1_data = THFloatTensor_data(xyz1);
    float *xyz2_data = THFloatTensor_data(xyz2);
    float *gradxyz1_data = THFloatTensor_data(gradxyz1);
    float *gradxyz2_data = THFloatTensor_data(gradxyz2);
    float *graddist1_data = THFloatTensor_data(graddist1);
    float *graddist2_data = THFloatTensor_data(graddist2);
    int *idx1_data = THIntTensor_data(idx1);
    int *idx2_data = THIntTensor_data(idx2);

    
    for (int i=0;i<b*n*3;i++)
        gradxyz1_data[i]=0;
    for (int i=0;i<b*m*3;i++)
        gradxyz2_data[i]=0;
    for (int i=0;i<b;i++){
        for (int j=0;j<n;j++){
            float x1=xyz1_data[(i*n+j)*3+0];
            float y1=xyz1_data[(i*n+j)*3+1];
            float z1=xyz1_data[(i*n+j)*3+2];
            int j2=idx1_data[i*n+j];

            float x2=xyz2_data[(i*m+j2)*3+0];
            float y2=xyz2_data[(i*m+j2)*3+1];
            float z2=xyz2_data[(i*m+j2)*3+2];
            float g=graddist1_data[i*n+j]*2;

            //printf("%d, %f\n", j2, g);

            gradxyz1_data[(i*n+j)*3+0]+=g*(x1-x2);
            gradxyz1_data[(i*n+j)*3+1]+=g*(y1-y2);
            gradxyz1_data[(i*n+j)*3+2]+=g*(z1-z2);
            gradxyz2_data[(i*m+j2)*3+0]-=(g*(x1-x2));
            gradxyz2_data[(i*m+j2)*3+1]-=(g*(y1-y2));
            gradxyz2_data[(i*m+j2)*3+2]-=(g*(z1-z2));
        }
        for (int j=0;j<m;j++){
            float x1=xyz2_data[(i*m+j)*3+0];
            float y1=xyz2_data[(i*m+j)*3+1];
            float z1=xyz2_data[(i*m+j)*3+2];
            int j2=idx2_data[i*m+j];
            float x2=xyz1_data[(i*n+j2)*3+0];
            float y2=xyz1_data[(i*n+j2)*3+1];
            float z2=xyz1_data[(i*n+j2)*3+2];
            float g=graddist2_data[i*m+j]*2;
            gradxyz2_data[(i*m+j)*3+0]+=g*(x1-x2);
            gradxyz2_data[(i*m+j)*3+1]+=g*(y1-y2);
            gradxyz2_data[(i*m+j)*3+2]+=g*(z1-z2);
            gradxyz1_data[(i*n+j2)*3+0]-=(g*(x1-x2));
            gradxyz1_data[(i*n+j2)*3+1]-=(g*(y1-y2));
            gradxyz1_data[(i*n+j2)*3+2]-=(g*(z1-z2));
        }
    }

    return 1;
}



