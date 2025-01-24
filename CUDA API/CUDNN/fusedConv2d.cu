#include <iostream> 
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cmath>
#include <vector>
#include <functional>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }

float benchmarkKernel(const std::function<void()>& kernelFunc,
                      int warmupRuns = 5,
                      int benchmarkRuns = 50) 
{
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    // Warmup runs
    for (int i = 0 ; i < warmupRuns ; i++) {
        kernelFunc();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // Timed Benchmark
    float totalMs = 0.f;
    for (int i = 0 ; i < benchmarkRuns ; i++) {\
        CHECK_CUDA(cudaEventRecord(start));

        kernelFunc();

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsedMs;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        totalMs += elapsedMs;
    }
    // Destroy events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Return average over benchmarkRuns
    return totalMs / benchmarkRuns;


}

int main() {
    // Declare dimensions
    int batchSize = 1;
    int inChannels = 3, outChannels = 8;
    int height = 224, width = 224;
    int kernelSize = 3, pad = 1, stride = 1, dilation = 1;
    std::cout << "Image size: " << width << "x" << height << "x" << inChannels << std::endl;
    std::cout << "Kernel size: " << kernelSize << "x" << kernelSize << "x" << inChannels << "x" << outChannels << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    
    // Derived output shapes
    int outHeight = (height + 2 * pad - dilation * (kernelSize - 1) - 1) / stride + 1;
    int outWidth  = (width  + 2 * pad - dilation * (kernelSize - 1) - 1) / stride + 1;

    // Host Allocations
    size_t inputSize = batchSize * inChannels * height * width;
    size_t kernelSize_ = outChannels * inChannels * kernelSize * kernelSize;
    size_t biasSize = outChannels;
    size_t outputSize = batchSize * outChannels * outHeight * outWidth;

    std::vector<float> h_input(inputSize), h_kernel(kernelSize_), h_bias(biasSize);
    std::vector<float> h_outputNonFused(outputSize), h_outputFused(outputSize);

    for (int i = 0; i < inputSize; i++) h_input[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < kernelSize_; i++) h_kernel[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < biasSize; i++) h_bias[i] = 0;

    // Allocate on device
    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_bias = nullptr;
    float* d_outNonFused = nullptr;
    float* d_outFused = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input,        inputSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel,       kernelSize_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias,         biasSize   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_outNonFused,  outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_outFused,     outputSize * sizeof(float)));

    // Copy data host->device
    CHECK_CUDA(cudaMemcpy(d_input,  h_input.data(),   inputSize*sizeof(float),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(),  kernelSize_*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias,   h_bias.data(),    biasSize*sizeof(float),    cudaMemcpyHostToDevice));

    // Zero output
    //CHECK_CUDA(cudaMemset(d_outNonFused, 0, outputSize*sizeof(float)));
    //CHECK_CUDA(cudaMemset(d_outFused,    0, outputSize*sizeof(float)));
    
    // Create handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Descriptors
    cudnnTensorDescriptor_t inDesc, outDesc, biasDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t actDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&biasDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                pad, pad,
                                                stride, stride,
                                                dilation, dilation,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    
    // Input Descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inChannels, height, width));

    // Output descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, outChannels, outHeight, outWidth));

    // Bias descriptor (1, outChannels, 1, 1)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, outChannels, 1, 1));

    // Activation descriptor RELU
    CHECK_CUDNN(cudnnSetActivationDescriptor(actDesc,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0 /*relu_coef*/));
        
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, 
                                            CUDNN_DATA_FLOAT,    
                                            CUDNN_TENSOR_NCHW,   
                                            outChannels,         
                                            inChannels,          
                                            kernelSize,          
                                            kernelSize));      

    // Choose the convolution algorithm and the workspace
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; 
    size_t workspaceSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        inDesc, kernelDesc, convDesc, outDesc,
        algo, &workspaceSize));

    // Allocate workspace
    void* d_workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    }

    // Benchmarking Params
    int warmupRuns = 5;
    int benchmarkRuns = 50;

    /* ******** Non Fused Operations *********** */
    float alpha = 1.f, beta = 0.f;
    float alphaBias = 1.f, betaBias = 1.f;
    float alphaAct = 1.f, betaAct = 0.f; 
    auto nonFusedOp = [&]() {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            inDesc, d_input,
            kernelDesc, d_kernel,
            convDesc, algo,
            d_workspace, workspaceSize,
            &beta,
            outDesc, d_outNonFused));

        CHECK_CUDNN(cudnnAddTensor(
            cudnn,
            &alphaBias,
            biasDesc, d_bias,
            &betaBias,
            outDesc, d_outNonFused));

        CHECK_CUDNN(cudnnActivationForward(
            cudnn, actDesc,
            &alphaAct,
            outDesc, d_outNonFused,
            &betaAct,
            outDesc, d_outNonFused
        ));
    };
    float avgNonFusedMs = benchmarkKernel(nonFusedOp, warmupRuns, benchmarkRuns);

    /* ******** Fused Operations *********** */
    CHECK_CUDA(cudaMemset(d_outFused, 0, outputSize*sizeof(float)));
    auto fusedOp = [&]() {
        CHECK_CUDA(cudaMemset(d_outFused, 0, outputSize*sizeof(float)));
        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            cudnn,
            &alpha,
            inDesc, d_input,
            kernelDesc, d_kernel,
            convDesc,
            algo,
            d_workspace, workspaceSize,
            &beta,
            outDesc, d_outFused,
            biasDesc, d_bias,
            actDesc,
            outDesc, d_outFused));
    };
    float avgFusedMs    = benchmarkKernel(fusedOp,    warmupRuns, benchmarkRuns);
    
    // Print Benchmark Results
    std::cout << "Non-fused average time: " << avgNonFusedMs << " ms\n";
    std::cout << "Fused average time:     " << avgFusedMs    << " ms\n";

    // Copy back data  Device -> Host
    CHECK_CUDA(cudaMemcpy(h_outputNonFused.data(), d_outNonFused,
                         outputSize*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_outputFused.data(), d_outFused,
                          outputSize*sizeof(float), cudaMemcpyDeviceToHost));  

    // Compute max difference
    float maxDiff = 0.0f;
    float meanDiff = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        float diff = fabs(h_outputNonFused[i] - h_outputFused[i]);
        meanDiff += diff;
        if (diff > maxDiff) maxDiff = diff;
    }
    std::cout << "Max difference (non-fused vs fused) = " << maxDiff << std::endl;
    std::cout << "Mean difference (non-fused vs fused) = " << meanDiff / outputSize << std::endl;

    // Deallocate
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_outNonFused));
    CHECK_CUDA(cudaFree(d_outFused));

    CHECK_CUDNN(cudnnDestroyActivationDescriptor(actDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernelDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(biasDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
    
}