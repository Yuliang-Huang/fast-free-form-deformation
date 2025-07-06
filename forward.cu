#include "forward.h"
#include "bspline_weight.h"

// Sample displacement field with cubic interpolation
__device__ void sampleDisplacementAndJacobian(
    cudaTextureObject_t tex, 
    int ix, 
    int iy, 
    int iz,
    float Bx[4], 
    float By[4], 
    float Bz[4],
    float dBx[4],
    float dBy[4],
    float dBz[4], 
    float& displacement, 
    float3& gradient
) 
{
    // Step 1: Interpolate along X (4x4x4 → 4x4)
    float col[4][4], col_dx[4][4];
    for (int dz = 0; dz < 4; dz++) 
    {
        for (int dy = 0; dy < 4; dy++) 
        {
            col[dz][dy] = Bx[0] * tex3D<float>(tex, ix - 0.5f, iy + dy - 0.5f, iz + dz - 0.5f)
                        + Bx[1] * tex3D<float>(tex, ix + 0.5f, iy + dy - 0.5f, iz + dz - 0.5f)
                        + Bx[2] * tex3D<float>(tex, ix + 1.5f, iy + dy - 0.5f, iz + dz - 0.5f)
                        + Bx[3] * tex3D<float>(tex, ix + 2.5f, iy + dy - 0.5f, iz + dz - 0.5f);
            col_dx[dz][dy] = dBx[0] * tex3D<float>(tex, ix - 0.5f, iy + dy - 0.5f, iz + dz - 0.5f)
                          + dBx[1] * tex3D<float>(tex, ix + 0.5f, iy + dy - 0.5f, iz + dz - 0.5f)
                          + dBx[2] * tex3D<float>(tex, ix + 1.5f, iy + dy - 0.5f, iz + dz - 0.5f)
                          + dBx[3] * tex3D<float>(tex, ix + 2.5f, iy + dy - 0.5f, iz + dz - 0.5f);
        }
    }

    // Step 2: Interpolate along Y (4x4 → 4)
    float row[4], row_dx[4], row_dy[4];
    for (int dz = 0; dz < 4; dz++) 
    {
        row[dz] = By[0] * col[dz][0] + By[1] * col[dz][1] + By[2] * col[dz][2] + By[3] * col[dz][3];
        row_dx[dz] = By[0] * col_dx[dz][0] + By[1] * col_dx[dz][1] + By[2] * col_dx[dz][2] + By[3] * col_dx[dz][3];
        row_dy[dz] = dBy[0] * col[dz][0] + dBy[1] * col[dz][1] + dBy[2] * col[dz][2] + dBy[3] * col[dz][3];
    }

    // Step 3: Interpolate along Z (4 → 1)
    displacement = Bz[0] * row[0] + Bz[1] * row[1] + Bz[2] * row[2] + Bz[3] * row[3];
    gradient.x = Bz[0] * row_dx[0] + Bz[1] * row_dx[1] + Bz[2] * row_dx[2] + Bz[3] * row_dx[3];
    gradient.y = Bz[0] * row_dy[0] + Bz[1] * row_dy[1] + Bz[2] * row_dy[2] + Bz[3] * row_dy[3];
    gradient.z = dBz[0] * row[0] + dBz[1] * row[1] + dBz[2] * row[2] + dBz[3] * row[3];
}


// Forward pass kernel
__global__ void forwardKernel(
    cudaTextureObject_t* texArray, 
    const float3* inputPoints, 
    float* outputDisplacements, 
    float3* outputJacobian, 
    int numPoints,
    int width, 
    int height, 
    int depth
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    int ichannel = blockIdx.y;

    float3 pos = inputPoints[idx];
    int ix = floorf(pos.x);
    int iy = floorf(pos.y);
    int iz = floorf(pos.z);
    float fx = pos.x - ix;
    float fy = pos.y - iy;
    float fz = pos.z - iz;
    if (ix < 1 || ix >= width - 2 || iy < 1 || iy >= height - 2 || iz < 1 || iz >= depth - 2) {
        return;
    }
    
    float Bx[4], dBx[4], By[4], dBy[4], Bz[4], dBz[4];
    computeBSplineWeights(fx, Bx);
    computeBSplineWeights(fy, By);
    computeBSplineWeights(fz, Bz);
    computeBSplineWeightsDerivative(fx, dBx);
    computeBSplineWeightsDerivative(fy, dBy);
    computeBSplineWeightsDerivative(fz, dBz);

    // Compute displacements and their gradients
    float displacement;
    float3 du_dpos;
    sampleDisplacementAndJacobian(texArray[ichannel], ix, iy, iz, Bx, By, Bz, dBx, dBy, dBz, displacement, du_dpos);

    outputDisplacements[idx*3+ichannel] = displacement;

    // Store Jacobian matrix in column-major order
    // Later in the rasterizer, we need J_transpose stored in row-major order
    outputJacobian[idx * 3 + ichannel] = du_dpos;
}

//overloaded forwardKernel for the case of no jacobian
__global__ void forwardKernel(
    cudaTextureObject_t * texArray, 
    const float3* inputPoints, 
    float* outputs, 
    int numPoints,
    int width, 
    int height, 
    int depth,
    int nchannels
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ichannel = blockIdx.y;
    if (idx >= numPoints) return;

    float3 pos = inputPoints[idx];
    int ix = floorf(pos.x);
    int iy = floorf(pos.y);
    int iz = floorf(pos.z);
    float fx = pos.x - ix;
    float fy = pos.y - iy;
    float fz = pos.z - iz;
    if (ix < 1 || ix >= width - 2 || iy < 1 || iy >= height - 2 || iz < 1 || iz >= depth - 2) {
        return;
    }
    
    float Bx[4], By[4], Bz[4];
    computeBSplineWeights(fx, Bx);
    computeBSplineWeights(fy, By);
    computeBSplineWeights(fz, Bz);
    
    // Step 1: Interpolate along X (4x4x4 → 4x4)
    float col[4][4];
    for (int dz = -1; dz <= 2; dz++) 
    {
        for (int dy = -1; dy <= 2; dy++) 
        {
            col[dz+1][dy+1] = Bx[0] * tex3D<float>(texArray[ichannel], ix - 0.5f, iy + dy + 0.5f, iz + dz + 0.5f)
                            + Bx[1] * tex3D<float>(texArray[ichannel], ix + 0.5f, iy + dy + 0.5f, iz + dz + 0.5f)
                            + Bx[2] * tex3D<float>(texArray[ichannel], ix + 1.5f, iy + dy + 0.5f, iz + dz + 0.5f)
                            + Bx[3] * tex3D<float>(texArray[ichannel], ix + 2.5f, iy + dy + 0.5f, iz + dz + 0.5f);
        }
    }

    // Step 2: Interpolate along Y (4x4 → 4)
    float row[4];
    for (int dz = 0; dz < 4; dz++) {
        row[dz] = By[0] * col[dz][0] + By[1] * col[dz][1] + By[2] * col[dz][2] + By[3] * col[dz][3];
    }

    // Step 3: Interpolate along Z (4 → 1)
    outputs[idx*nchannels+ichannel] = Bz[0] * row[0] + Bz[1] * row[1] + Bz[2] * row[2] + Bz[3] * row[3];
       
}