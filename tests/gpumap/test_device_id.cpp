// Test for src/acc/device_id.{h,cpp}.
// CUDA:   nvcc -x cu -D_CUDA_ENABLED -I. -o t tests/gpumap/test_device_id.cpp -x cu src/acc/device_id.cpp -lcuda
// no GPU: g++ -std=c++11 -I. -o t tests/gpumap/test_device_id.cpp src/acc/device_id.cpp   (env and index sources only)
#include "src/acc/device_id.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#ifdef _CUDA_ENABLED
#include <cuda_runtime.h>
#endif

// a source that always answers, to test ordering and preference
class FixedSource : public DeviceIdSource
{
public:
	FixedSource(const std::string &n, const std::string &v) : n(n), v(v) {}
	std::string name() const { return n; }
	std::string identify(int) const { return v; }
private:
	std::string n, v;
};

int main()
{
	// helpers
	assert(EnvironmentDeviceIdSource::looksLikeUuid("GPU-8b5e715e-0a57-8390-3ed7-750bf7480d9a"));
	assert(EnvironmentDeviceIdSource::looksLikeUuid("MIG-0d060d94-8b87-5ea2-8854-16d3c4bbfa3f"));
	assert(!EnvironmentDeviceIdSource::looksLikeUuid("0"));
	assert(EnvironmentDeviceIdSource::splitList("a, b ,c")[1] == "b");

	// environment source: indices are not identities, UUID lists and explicit lists are
	unsetenv("RELION_GPU_UIDS");
	setenv("TEST_VISIBLE", "0,1", 1);
	assert(EnvironmentDeviceIdSource(2, "TEST_VISIBLE").identify(1).empty());
	setenv("TEST_VISIBLE", "GPU-8b5e715e-0a57-8390-3ed7-750bf7480d9a,MIG-0d060d94-8b87-5ea2-8854-16d3c4bbfa3f", 1);
	assert(EnvironmentDeviceIdSource(2, "TEST_VISIBLE").identify(1) == "env:MIG-0d060d94-8b87-5ea2-8854-16d3c4bbfa3f");
	assert(EnvironmentDeviceIdSource(3, "TEST_VISIBLE").identify(1).empty());   // count mismatch
	setenv("RELION_GPU_UIDS", "site-a,site-b", 1);
	assert(EnvironmentDeviceIdSource(2, "TEST_VISIBLE").identify(0) == "env:site-a");
	unsetenv("RELION_GPU_UIDS");

	// resolver: order, preference, fallback
	DeviceIdResolver r;
	r.add(std::unique_ptr<DeviceIdSource>(new FixedSource("a", "")));
	r.add(std::unique_ptr<DeviceIdSource>(new FixedSource("b", "B")));
	r.add(std::unique_ptr<DeviceIdSource>(new FixedSource("c", "C")));
	assert(r.identify(0) == "B");
	r.preferSource("c");
	assert(r.identify(0) == "C");
	assert(IndexDeviceIdSource("h").identify(3) == "h/index:3");

	// real devices
	int n = 0;
#ifdef _CUDA_ENABLED
	if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
#endif
	const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
	std::cout << "devices=" << n << " CUDA_VISIBLE_DEVICES=" << (cvd ? cvd : "unset") << "\n";
	const char *modes[] = {"uuid", "env", "pci", "index"};
	for (int d = 0; d < n; d++)
	{
		unsetenv("RELION_GPU_UID_SOURCE");
		std::cout << "dev" << d << " default -> " << DeviceIdResolver::forThisBackend(n, "host1").identify(d) << "\n";
		for (int m = 0; m < 4; m++)
		{
			setenv("RELION_GPU_UID_SOURCE", modes[m], 1);
			std::cout << "dev" << d << " prefer " << modes[m] << " -> " << DeviceIdResolver::forThisBackend(n, "host1").identify(d) << "\n";
		}
	}
	std::cout << "OK\n";
	return 0;
}
