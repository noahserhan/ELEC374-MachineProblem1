
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

int getNumCores(cudaDeviceProp dev);

int main(void) {
	int devices;
	cudaGetDeviceCount(&devices);
	
	for (int i = 0; i < devices; i++) {
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);

		std::cout << "Device Name: " << properties.name << std::endl;
		std::cout << "Clock Rate: " << properties.clockRate << std::endl;
		std::cout << "Number of Streaming Multiprocessors: " << properties.multiProcessorCount << std::endl;
		std::cout << "Number of CUDA Cores: " << getNumCores(properties) << std::endl;
		std::cout << "Warp Size: " << properties.warpSize << std::endl;
		std::cout << "Global Memory Size: " << properties.totalGlobalMem << std::endl;
		std::cout << "Constant Memory Size: " << properties.totalConstMem << std::endl;
		std::cout << "Shared Memory Per Block: " << properties.sharedMemPerBlock << std::endl;
		std::cout << "Number of Registers Per Block: " << properties.regsPerBlock << std::endl;
		std::cout << "Maximum Number of Threads Per Block: " << properties.maxThreadsPerBlock << std::endl;
		std::cout << "Maximum Size of Each Dimension of a Block: (" << properties.maxThreadsDim[0] << ", "
			<< properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Maximum Size of Each Dimension of a Grid: (" << properties.maxGridSize[0] << ", "
			<< properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << ")" << std::endl;
	}
}

int getNumCores(cudaDeviceProp dev) {
	int cores = 0;
	int smCount = dev.multiProcessorCount;

	switch (dev.major) {
		// Depending on the relevant GPU architecture
	case 2:
		if (dev.minor == 1) cores = smCount * 48;
		else cores = smCount * 32;
		break;
	case 3:
		cores = smCount * 192;
		break;
	case 5:
		cores = smCount * 128;
		break;
	case 6:
		if ((dev.minor == 1) || (dev.minor == 2)) cores = smCount * 128;
		else if (dev.minor == 0) cores = smCount * 64;
		else std::cout << "Unknown architecture compute capability\n" << std::endl;
		break;
	case 7:
		if ((dev.minor == 1) || (dev.minor == 5)) cores = smCount * 128;
		else if (dev.minor == 0) cores = smCount * 64;
		else std::cout << "Unknown architecture compute capability\n" << std::endl;
		break;
	case 8:
		if (dev.minor == 0) cores = smCount * 64;
		else if (dev.minor == 6) cores = smCount * 128;
		else if (dev.minor == 9) cores = smCount * 128;
		else std::cout << "Unknown architecture compute capability\n" << std::endl;
		break;
	case 9:
		if (dev.minor == 0) cores = smCount * 128;
		else std::cout << "Unknown architecture compute capability\n" << std::endl;
		break;
	default:
		std::cout << "Unknown architecture compute capability\n" << std::endl;
		break;
	}
	return cores;
}
