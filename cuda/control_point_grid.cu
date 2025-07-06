#include "control_point_grid.h"

// Constructor
ControlPointGrid::ControlPointGrid(int w, int h, int d, int nchannels) 
{
    this->width = w;
    this->height = h;
    this->depth = d;
    this->nchannels = nchannels;

    cudaMallocManaged(&this->texArray, nchannels * sizeof(cudaTextureObject_t));//https://forums.developer.nvidia.com/t/cuda-passing-a-class-to-a-kernel/219779/2
    this->dataArray = new cudaArray*[nchannels];

    for (int i = 0; i<nchannels; i++)
    {
        this->create3DTexture(this->dataArray[i], this->texArray[i]);
    }
}

// Destructor
ControlPointGrid::~ControlPointGrid() 
{
    for (int i = 0; i<this->nchannels; i++)
    {
        cudaDestroyTextureObject(this->texArray[i]);
        cudaFreeArray(this->dataArray[i]);            
    }
}

// Uploads device data to 3D texture
void ControlPointGrid::copyData(torch::Tensor &data) {

    if (data.size(0)!=this->nchannels || data.size(1)!=this->depth || data.size(2)!=this->height || data.size(3)!=this->width)
    {
        std::cerr << "ERROR input tensor shape mismatched with Texture Container extent!" << std::endl;
    }

    cudaMemcpy3DParms copyParams = {0};

    // Upload X displacement from device memory
    copyParams.extent = make_cudaExtent(this->width, this->height, this->depth);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    for (int i = 0; i<this->nchannels; i++)
    {
        copyParams.srcPtr = make_cudaPitchedPtr(data.index({i}).data_ptr<float>(), this->width * sizeof(float), this->width, this->height);
        copyParams.dstArray = this->dataArray[i];
        cudaMemcpy3D(&copyParams);
    }
}

// Creates a 3D texture and surface object
void ControlPointGrid::create3DTexture(cudaArray_t& d_array, cudaTextureObject_t& tex) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(this->width, this->height, this->depth);
    
    // Allocate 3D CUDA array
    cudaMalloc3DArray(&d_array, &channelDesc, volumeSize);

    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    cudaTextureDesc texDesc = {};
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}
