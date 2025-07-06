#include "backward.h"
#include "bspline_weight.h"

__global__ void backwardKernel(
    cudaTextureObject_t texX, 
    cudaTextureObject_t texY, 
    cudaTextureObject_t texZ,
    const float3* inputPoints,  
    float3* gradControlPoints,
    const float3* gradDisplacements, 
    const float* gradJacobians, 
    int numPoints,
    int width, 
    int height, 
    int depth
)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // Load input point
    float3 p = inputPoints[idx];

    // Integer base index and fractional offsets
    int ix = floorf(p.x);
    int iy = floorf(p.y);
    int iz = floorf(p.z);

    if (ix < 1 || ix >= width - 2 || iy < 1 || iy >= height - 2 || iz < 1 || iz >= depth - 2) {
        return;
    }

    float fx = p.x - ix;
    float fy = p.y - iy;
    float fz = p.z - iz;

    // Compute B-spline weights and derivatives
    float Bx[4], dBx[4], d2Bx[4];
    float By[4], dBy[4], d2By[4];
    float Bz[4], dBz[4], d2Bz[4];

    computeBSplineWeights(fx, Bx);
    computeBSplineWeights(fy, By);
    computeBSplineWeights(fz, Bz);
    computeBSplineWeightsDerivative(fx, dBx);
    computeBSplineWeightsDerivative(fy, dBy);
    computeBSplineWeightsDerivative(fz, dBz);
    computeBSplineWeightsSecondDerivative(fx, d2Bx);
    computeBSplineWeightsSecondDerivative(fy, d2By);
    computeBSplineWeightsSecondDerivative(fz, d2Bz);

    // Load gradients from output
    const float3 gradD = gradDisplacements[idx]; // ∂L/∂d
    float gradJ[3][3] = {0.0f};
    if (gradJacobians != nullptr) {
        for (int i = 0; i < 9; i++) {
            gradJ[i / 3][i % 3] = gradJacobians[idx * 9 + i]; // ∂L/∂J
        }
    }
    else
    {
        printf("gradJacobians is nullptr at idx %d\n",idx);
    }

    for (int dz = 0; dz < 4; dz++) 
    {
        for (int dy = 0; dy < 4; dy++) 
        {
            for (int dx = 0; dx < 4; dx++) 
            {
                float weight = Bx[dx] * By[dy] * Bz[dz];
                float dWeight_x = dBx[dx] * By[dy] * Bz[dz];
                float dWeight_y = Bx[dx] * dBy[dy] * Bz[dz];
                float dWeight_z = Bx[dx] * By[dy] * dBz[dz];

                float d2Weight_xx = d2Bx[dx] * By[dy] * Bz[dz];
                float d2Weight_yy = Bx[dx] * d2By[dy] * Bz[dz];
                float d2Weight_zz = Bx[dx] * By[dy] * d2Bz[dz];

                float d2Weight_xy = dBx[dx] * dBy[dy] * Bz[dz];
                float d2Weight_xz = dBx[dx] * By[dy] * dBz[dz];
                float d2Weight_yz = Bx[dx] * dBy[dy] * dBz[dz];

                // Sample control point values
                // should start from minus 1 and then shift 0.5 to get to the voxel center
                float vx = tex3D<float>(texX, ix + dx - 0.5f, iy + dy - 0.5f, iz + dz - 0.5f);
                float vy = tex3D<float>(texY, ix + dx - 0.5f, iy + dy - 0.5f, iz + dz - 0.5f);
                float vz = tex3D<float>(texZ, ix + dx - 0.5f, iy + dy - 0.5f, iz + dz - 0.5f);

                // Backprop to control points (∂L/∂C=∂L/∂d * ∂d/∂C + ∂L/∂J * ∂J/∂C)
                int offset = ((ix + dx - 1) * height + (iy + dy - 1)) * depth + (iz + dz - 1);
                atomicAdd(&gradControlPoints[offset].x, weight * gradD.x + dWeight_x * gradJ[0][0] + dWeight_y * gradJ[0][1] + dWeight_z * gradJ[0][2]);
                atomicAdd(&gradControlPoints[offset].y, weight * gradD.y + dWeight_x * gradJ[1][0] + dWeight_y * gradJ[1][1] + dWeight_z * gradJ[1][2]);
                atomicAdd(&gradControlPoints[offset].z, weight * gradD.z + dWeight_x * gradJ[2][0] + dWeight_y * gradJ[2][1] + dWeight_z * gradJ[2][2]);
            }
        }
    }
}

//overloaded backward pass kernel
__global__ void backwardKernel(
    cudaTextureObject_t * texArray, 
    const float3* inputPoints,
    float* gradControlPoints,
    const float* gradOutputs,
    int numPoints,
    int width, 
    int height, 
    int depth,
    int nchannels
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // Load input point
    float3 p = inputPoints[idx];

    // Integer base index and fractional offsets
    int ix = floorf(p.x);
    int iy = floorf(p.y);
    int iz = floorf(p.z);

    if (ix < 1 || ix >= width - 2 || iy < 1 || iy >= height - 2 || iz < 1 || iz >= depth - 2) {
        return;
    }

    float fx = p.x - ix;
    float fy = p.y - iy;
    float fz = p.z - iz;

    // Compute B-spline weights and derivatives
    float Bx[4], dBx[4];
    float By[4], dBy[4];
    float Bz[4], dBz[4];

    computeBSplineWeights(fx, Bx);
    computeBSplineWeights(fy, By);
    computeBSplineWeights(fz, Bz);
    computeBSplineWeightsDerivative(fx, dBx);
    computeBSplineWeightsDerivative(fy, dBy);
    computeBSplineWeightsDerivative(fz, dBz);

    // Load gradients from output
    const float * gradOutput = gradOutputs + idx * nchannels; // ∂L/∂d

    // Initialize gradients
    float3 gradP = make_float3(0, 0, 0);

    // Backpropagate through control points
    for (int dz = 0; dz < 4; dz++) 
    {
        for (int dy = 0; dy < 4; dy++) 
        {
            for (int dx = 0; dx < 4; dx++) 
            {
                float weight = Bx[dx] * By[dy] * Bz[dz];
                float dWeight_x = dBx[dx] * By[dy] * Bz[dz];
                float dWeight_y = Bx[dx] * dBy[dy] * Bz[dz];
                float dWeight_z = Bx[dx] * By[dy] * dBz[dz];
                int offset = (((ix + dx - 1) * height + (iy + dy - 1)) * depth + (iz + dz - 1))*nchannels;
                for (int i = 0; i < nchannels; i++) 
                {
                    // Backprop to control points (∂L/∂C=∂L/∂O * ∂O/∂C)
                    atomicAdd(&gradControlPoints[offset+i], weight*gradOutput[i]);
                }
            }
        }
    }
}