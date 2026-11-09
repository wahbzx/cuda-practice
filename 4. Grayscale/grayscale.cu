#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

bool saveImage(const uint8_t* pixels, int width, int height, int channels) {
    return stbi_write_png("output.png", width, height, channels, pixels, width * channels) != 0;
}

void grayscaleCPU(uint8_t* pixelData, int w, int h, int c) {
    for (int i = 0; i < w * h; ++i) {
        int offset = i * c;
        // Using Rec. 709 formula for converting colour to grayscale https://en.wikipedia.org/wiki/Grayscale
        uint8_t gray = static_cast<uint8_t>(
            (0.2126f * pixelData[offset] + 0.7152f * pixelData[offset+ 1] + 0.0722f * pixelData[offset + 2]));
        pixelData[offset] = gray;
        pixelData[offset+ 1] = gray;
        pixelData[offset+ 2] = gray;
    }
}


__global__
void grayscaleGPU(uint8_t* pixelData, int w, int h, int c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < h && col < w) {
        int offset = (row * w + col) * c;
        uint8_t gray = static_cast<uint8_t>(
            (0.2126f * pixelData[offset] + 0.7152f * pixelData[offset + 1] + 0.0722f * pixelData[offset + 2]));
        pixelData[offset] = gray;
        pixelData[offset+ 1] = gray;
        pixelData[offset+ 2] = gray;
    }
}

int main() {
    int width, height, channels, req_channels;
    req_channels = 3;
    
    uint8_t* pixelData = stbi_load("image.png", &width, &height, &channels, req_channels);
    if (pixelData == nullptr) {
        std::cout << "Failed to load image!" << std::endl;
        return -1;
    }
    size_t imageBytes = static_cast<size_t>(width) * height * req_channels;

    uint8_t* pixelData_d;
    cudaMalloc(&pixelData_d, imageBytes);
    cudaMemcpy(pixelData_d, pixelData, imageBytes, cudaMemcpyHostToDevice);

    // Our image is 256x256 so we process it in 32x32 blocks on an 8x8 grid
    // TODO: Probably should make the calculation dynamic based on image size
    dim3 dimGrid(8, 8);
    dim3 dimBlock(32, 32);
    
    grayscaleGPU<<<dimGrid, dimBlock>>>(pixelData_d, width, height, req_channels);

    cudaMemcpy(pixelData, pixelData_d, imageBytes, cudaMemcpyDeviceToHost);
    // Immediately free GPU memory
    cudaFree(pixelData_d);
    
    // Save image
    if (!saveImage(pixelData, width, height, req_channels)) {
        std::cerr << "Failed to save output.png" << std::endl;
        stbi_image_free(pixelData);
        return -1;
    }

    // Free everything
    stbi_image_free(pixelData); 
    return 0;
}
