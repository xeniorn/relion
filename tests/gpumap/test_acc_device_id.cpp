// Test for src/acc/acc_device_id.h.
// Build (CUDA):   nvcc -x cu -D_CUDA_ENABLED -I. -o t tests/gpumap/test_acc_device_id.cpp -lcuda
// Build (no GPU): g++ -std=c++11 -I. -o t tests/gpumap/test_acc_device_id.cpp   (env/index sources only)
#include "src/acc/acc_device_id.h"
#include <cassert>
#include <iostream>
using namespace acc_device_id_detail;

int main()
{
	// helpers
	assert(looksLikeUuid("GPU-8b5e715e-0a57-8390-3ed7-750bf7480d9a"));
	assert(looksLikeUuid("MIG-0d060d94-8b87-5ea2-8854-16d3c4bbfa3f"));
	assert(!looksLikeUuid("0"));
	assert(!looksLikeUuid("0,1"));
	assert(splitList("a, b ,c").size() == 3 && splitList("a, b ,c")[1] == "b");

	int n = 0;
#ifdef _CUDA_ENABLED
	if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
#endif
	const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
	std::cout << "devices=" << n << " CUDA_VISIBLE_DEVICES=" << (cvd ? cvd : "unset") << "\n";
	const char *modes[] = {"auto", "api", "env", "pci", "index"};
	for (int d = 0; d < (n ? n : 1); d++)
		for (int m = 0; m < 5; m++)
		{
			setenv("RELION_GPU_UID_SOURCE", modes[m], 1);
			std::cout << "dev" << d << " source=" << modes[m] << " -> " << getAccPersistentDeviceId(d, n, "host1") << "\n";
		}
	// explicit override list
	setenv("RELION_GPU_UID_SOURCE", "env", 1);
	setenv("RELION_GPU_UIDS", "site-gpu-A", 1);
	std::cout << "RELION_GPU_UIDS=site-gpu-A, 1 device -> " << getAccPersistentDeviceId(0, 1, "host1") << "\n";
	std::cout << "RELION_GPU_UIDS count mismatch (2 devices) -> " << getAccPersistentDeviceId(0, 2, "host1") << "\n";
	std::cout << "OK\n";
	return 0;
}
