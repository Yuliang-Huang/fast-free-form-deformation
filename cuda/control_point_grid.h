#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

class ControlPointGrid {
public:
    ControlPointGrid(int w, int h, int d, int nchannels);
    ~ControlPointGrid();

    void copyData(torch::Tensor &data);    

    int width, height, depth, nchannels;
    // cudaTextureObject_t texX, texY, texZ;
    cudaTextureObject_t * texArray;

private:
    // cudaArray_t d_arrayX, d_arrayY, d_arrayZ;
    cudaArray ** dataArray;

    void create3DTexture(cudaArray_t& d_array, cudaTextureObject_t& tex);
};
