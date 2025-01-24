#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <functional>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) {                                              \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                        \
    }                                                                   \
}

#define CHECK_CUDNN(call) {                                             \
    cudnnStatus_t err = call;                                           \
    if (err != CUDNN_STATUS_SUCCESS) {                                  \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(err)        \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                        \
    }                                                                   \
}

// ---------------------------------------------------------------------
//  benchmarking 
// ---------------------------------------------------------------------
float benchmarkKernel(const std::function<void()>& kernelFunc,
                      int warmupRuns = 5,
                      int benchmarkRuns = 50)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < warmupRuns; i++) {
        kernelFunc();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Timed runs
    float totalMs = 0.f;
    for (int i = 0; i < benchmarkRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));

        kernelFunc();

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsedMs;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        totalMs += elapsedMs;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Average
    return totalMs / benchmarkRuns;
}

// ---------------------------------------------------------------------
//  create a Tensor descriptor
// ---------------------------------------------------------------------
cudnnTensorDescriptor_t createTensorDescriptor(
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int n, int c, int h, int w)
{
    cudnnTensorDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, format, dataType, n, c, h, w));
    return desc;
}

int main() {
    // Problem Size 
    const int width = 224;
    const int height = 224;
    const int kernelSize = 11;
    const int inChannels = 32;
    const int outChannels = 64;
    const int batchSize = 32;
    const int pad = kernelSize / 2;
    const int dilation = 1;
    const int stride = 1;
    int outHeight = (height + 2 * pad - dilation * (kernelSize - 1) - 1) / stride + 1;
    int outWidth  = (width  + 2 * pad - dilation * (kernelSize - 1) - 1) / stride + 1;


    size_t inputSize   = (size_t)batchSize * inChannels * height * width;
    size_t kernelSize_ = (size_t)outChannels * inChannels * kernelSize * kernelSize;
    size_t biasSize    = (size_t)outChannels;
    size_t outputSize  = (size_t)batchSize * outChannels * outHeight * outWidth;

    // Host data in FP32
    std::vector<float> h_input_fp32(inputSize);
    std::vector<float> h_kernel_fp32(kernelSize_);
    std::vector<float> h_bias_fp32(biasSize);
    for(int i=0; i<inputSize; i++) h_input_fp32[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i=0; i<kernelSize_; i++) h_kernel_fp32[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i=0; i<biasSize; i++) h_bias_fp32[i] = 0.1f;

    std::vector<__half> h_input_fp16(inputSize);
    std::vector<__half> h_kernel_fp16(kernelSize_);
    std::vector<__half> h_bias_fp16(biasSize);
    for(int i=0; i<inputSize; i++) h_input_fp16[i] = __float2half(h_input_fp32[i]);
    for(int i=0; i<kernelSize_; i++) h_kernel_fp16[i] = __float2half(h_kernel_fp32[i]);
    for(int i=0; i<biasSize; i++) h_bias_fp16[i] = __float2half(h_bias_fp32[i]);

    // Allocate GPU Memory
    // FP32
    float * d_input_fp32 = nullptr;
    float * d_kernel_fp32 = nullptr;
    float * d_bias_fp32 = nullptr;
    float * d_output_fp32 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input_fp32, inputSize      * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel_fp32, kernelSize_ *    sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias_fp32, biasSize *        sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_fp32,   outputSize  * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input_fp32, h_input_fp32.data(), inputSize * sizeof(float),    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel_fp32, h_kernel_fp32.data(), kernelSize_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias_fp32, h_bias_fp32.data(), biasSize * sizeof(float),      cudaMemcpyHostToDevice));
    
    // FP16
    __half* d_input_fp16  = nullptr;
    __half* d_kernel_fp16 = nullptr;
    __half* d_bias_fp16   = nullptr;
    __half* d_output_fp16 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input_fp16,   inputSize   * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_kernel_fp16,  kernelSize_ * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_bias_fp16,    biasSize    * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_output_fp16,  outputSize  * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_input_fp16,  h_input_fp16.data(),  inputSize*sizeof(__half),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel_fp16, h_kernel_fp16.data(), kernelSize_*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias_fp16,   h_bias_fp16.data(),   biasSize*sizeof(__half),    cudaMemcpyHostToDevice));
    // CUDNN Setup
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    cudnnTensorDescriptor_t inDesc_fp32 = createTensorDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                                 batchSize, inChannels, height, width);
    cudnnTensorDescriptor_t outDesc_fp32 = createTensorDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                                 batchSize, outChannels, outHeight, outWidth);
    cudnnTensorDescriptor_t biasDesc_fp32 = createTensorDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                                   1, outChannels, 1, 1);
    cudnnFilterDescriptor_t filterDesc_fp32;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc_fp32));               
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc_fp32,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW,
                                            outChannels, inChannels, kernelSize, kernelSize));
    cudnnConvolutionDescriptor_t convDesc_fp32;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc_fp32));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc_fp32,
                                                pad, pad, stride, stride,
                                                dilation, dilation,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));  
    cudnnActivationDescriptor_t actDesc_fp32;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc_fp32));
    CHECK_CUDNN(cudnnSetActivationDescriptor(actDesc_fp32, 
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0                ));
    cudnnConvolutionFwdAlgo_t algo_fp32 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // Workspace for FP32
    size_t workspaceSize_fp32 = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        inDesc_fp32, filterDesc_fp32, convDesc_fp32, outDesc_fp32,
        algo_fp32, &workspaceSize_fp32));

    void* d_workspace_fp32 = nullptr;
    if (workspaceSize_fp32 > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace_fp32, workspaceSize_fp32));
    }
    // Workspace for FP16
    cudnnTensorDescriptor_t inDesc_fp16  = createTensorDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                                                  batchSize, inChannels, height, width);
    cudnnTensorDescriptor_t outDesc_fp16 = createTensorDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                                                  batchSize, outChannels, outHeight, outWidth);
    cudnnTensorDescriptor_t biasDesc_fp16 = createTensorDescriptor(CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                                                   1, outChannels, 1, 1);

    cudnnFilterDescriptor_t filterDesc_fp16;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc_fp16));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc_fp16,
                                           CUDNN_DATA_HALF,
                                           CUDNN_TENSOR_NCHW,
                                           outChannels, inChannels, kernelSize, kernelSize));

    cudnnConvolutionDescriptor_t convDesc_fp16;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc_fp16));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc_fp16,
                                                pad, pad, stride, stride,
                                                dilation, dilation,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT)); 
    cudnnSetConvolutionMathType(convDesc_fp16, CUDNN_TENSOR_OP_MATH);
    cudnnActivationDescriptor_t actDesc_fp16;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc_fp16));
    CHECK_CUDNN(cudnnSetActivationDescriptor(actDesc_fp16,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0));

    cudnnConvolutionFwdAlgoPerf_t perf;
    int returnedAlgoCount = 0;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        inDesc_fp16, filterDesc_fp16, convDesc_fp16, outDesc_fp16,
        1, // request 1 best algorithm
        &returnedAlgoCount,
        &perf
    ));
    auto algo_fp16 = perf.algo;

    size_t workspaceSize_fp16 = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        inDesc_fp16, filterDesc_fp16, convDesc_fp16, outDesc_fp16,
        algo_fp16, &workspaceSize_fp16));

    void* d_workspace_fp16 = nullptr;
    std::cout << "workspaceSize_fp16 = " << workspaceSize_fp16 << std::endl;

    if (workspaceSize_fp16 > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace_fp16, workspaceSize_fp16));
    }
 
    // Perform the Benchmark
    float alpha = 1.f, beta = 0.f;
    float alphaBias = 1.f, betaBias = 1.f;
    float alphaAct  = 1.f, betaAct  = 0.f;
    auto fp32_op = [&]() {
        CHECK_CUDA(cudaMemset(d_output_fp32, 0, outputSize * sizeof(float)));
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            inDesc_fp32, d_input_fp32,
            filterDesc_fp32, d_kernel_fp32,
            convDesc_fp32,
            algo_fp32,
            d_workspace_fp32, workspaceSize_fp32,
            &beta,
            outDesc_fp32, d_output_fp32));
        CHECK_CUDNN(cudnnAddTensor(
            cudnn,
            &alphaBias,
            biasDesc_fp32, d_bias_fp32,
            &betaBias,
            outDesc_fp32, d_output_fp32
        ));
        CHECK_CUDNN(cudnnActivationForward(
            cudnn,
            actDesc_fp32,
            &alphaAct,
            outDesc_fp32, d_output_fp32,
            &betaAct,
            outDesc_fp32, d_output_fp32));
    };
    auto fp16_op = [&]() {
        CHECK_CUDA(cudaMemset(d_output_fp16, 0, outputSize * sizeof(__half)));

        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            inDesc_fp16, d_input_fp16,
            filterDesc_fp16, d_kernel_fp16,
            convDesc_fp16,
            algo_fp16,
            d_workspace_fp16, workspaceSize_fp16,
            &beta,
            outDesc_fp16, d_output_fp16));

        CHECK_CUDNN(cudnnAddTensor(
            cudnn,
            &alphaBias,
            biasDesc_fp16, d_bias_fp16,
            &betaBias,
            outDesc_fp16, d_output_fp16));

        CHECK_CUDNN(cudnnActivationForward(
            cudnn,
            actDesc_fp16,
            &alphaAct,
            outDesc_fp16, d_output_fp16,
            &betaAct,
            outDesc_fp16, d_output_fp16));
    };

    // Run the Benchmark
    int warmupRuns = 5;
    int benchmarkRuns = 50;

    float timeFP32 = benchmarkKernel(fp32_op, warmupRuns, benchmarkRuns);
    float timeFP16 = benchmarkKernel(fp16_op, warmupRuns, benchmarkRuns);

    std::cout << "Average FP32 time: " << timeFP32 << " ms\n";
    std::cout << "Average FP16 time: " << timeFP16 << " ms\n";
    
    // Perform Difference Statistics 
    std::vector<float> h_out_fp32(outputSize), h_out_fp16_in_fp32(outputSize);
    CHECK_CUDA(cudaMemcpy(h_out_fp32.data(), d_output_fp32, outputSize*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<__half> h_out_fp16(outputSize);
    CHECK_CUDA(cudaMemcpy(h_out_fp16.data(), d_output_fp16, outputSize*sizeof(__half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < outputSize; i++) h_out_fp16_in_fp32[i] = __half2float(h_out_fp16[i]);
    double maxDiff = 0.0, sumDiff = 0.0;
    for (size_t i = 0; i < outputSize; i++) {
        float diff = std::fabs(h_out_fp32[i] - h_out_fp16_in_fp32[i]);
        maxDiff   = std::max(maxDiff, (double)diff);
        sumDiff  += diff;
    }
    std::cout << "FP16 vs FP32 => maxDiff: " << maxDiff
              << ", meanDiff: " << (sumDiff / outputSize) << std::endl;
}