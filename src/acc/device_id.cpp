#include "src/acc/device_id.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#ifdef _CUDA_ENABLED
	#include <cuda.h>
	#include <cuda_runtime.h>
#elif _HIP_ENABLED
	#include <hip/hip_runtime.h>
#endif

namespace
{
	std::string formatUuid(const unsigned char *b)
	{
		char s[48];
		snprintf(s, sizeof s, "uuid:%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
		         b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
		return s;
	}

#ifdef _CUDA_ENABLED
	/// CUDA driver UUID. cuDeviceGetUuid_v2 returns the MIG instance's UUID; the v1 call would
	/// return the parent GPU's UUID for every MIG slice of that GPU.
	class CudaUuidDeviceIdSource : public DeviceIdSource
	{
	public:
		std::string name() const { return "uuid"; }
		std::string identify(int deviceIndex) const
		{
			CUdevice device;
			CUuuid uuid;
			if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, deviceIndex) != CUDA_SUCCESS)
				return "";
	#if CUDA_VERSION >= 11040
			if (cuDeviceGetUuid_v2(&uuid, device) != CUDA_SUCCESS)
	#else
			if (cuDeviceGetUuid(&uuid, device) != CUDA_SUCCESS)
	#endif
				return "";
			return formatUuid(reinterpret_cast<const unsigned char *>(uuid.bytes));
		}
	};

	class CudaPciDeviceIdSource : public DeviceIdSource
	{
	public:
		explicit CudaPciDeviceIdSource(const std::string &hostName) : hostName(hostName) {}
		std::string name() const { return "pci"; }
		std::string identify(int deviceIndex) const
		{
			char bus[64] = {0};
			if (cudaDeviceGetPCIBusId(bus, sizeof bus, deviceIndex) != cudaSuccess)
				return "";
			return hostName + "/pci:" + bus;
		}
	private:
		std::string hostName;
	};
#elif _HIP_ENABLED
	class HipUuidDeviceIdSource : public DeviceIdSource
	{
	public:
		std::string name() const { return "uuid"; }
		std::string identify(int deviceIndex) const
		{
			hipDevice_t device;
			hipUUID uuid;
			if (hipDeviceGet(&device, deviceIndex) != hipSuccess || hipDeviceGetUuid(&uuid, device) != hipSuccess)
				return "";
			return formatUuid(reinterpret_cast<const unsigned char *>(uuid.bytes));
		}
	};

	class HipPciDeviceIdSource : public DeviceIdSource
	{
	public:
		explicit HipPciDeviceIdSource(const std::string &hostName) : hostName(hostName) {}
		std::string name() const { return "pci"; }
		std::string identify(int deviceIndex) const
		{
			char bus[64] = {0};
			if (hipDeviceGetPCIBusId(bus, sizeof bus, deviceIndex) != hipSuccess)
				return "";
			return hostName + "/pci:" + bus;
		}
	private:
		std::string hostName;
	};
#endif
}

// ------------------------------------------------------------------ EnvironmentDeviceIdSource

EnvironmentDeviceIdSource::EnvironmentDeviceIdSource(int deviceCount, const std::string &visibleDevicesVariable)
	: deviceCount(deviceCount), visibleDevicesVariable(visibleDevicesVariable)
{
}

std::string EnvironmentDeviceIdSource::identify(int deviceIndex) const
{
	const char *explicitList = getenv("RELION_GPU_UIDS");
	const char *visible = getenv(visibleDevicesVariable.c_str());
	const char *list = explicitList ? explicitList : visible;
	if (list == NULL)
		return "";

	const std::vector<std::string> ids = splitList(list);
	if ((int)ids.size() != deviceCount || deviceIndex < 0 || deviceIndex >= deviceCount)
		return "";

	// an explicit list is trusted as given; the visible-devices variable only if it holds UUIDs
	// (plain indices like "0,1" say nothing about the physical device)
	if (explicitList == NULL)
		for (size_t i = 0; i < ids.size(); i++)
			if (!looksLikeUuid(ids[i]))
				return "";

	return ids[deviceIndex].empty() ? "" : "env:" + ids[deviceIndex];
}

bool EnvironmentDeviceIdSource::looksLikeUuid(const std::string &text)
{
	std::string u = text;
	if (u.compare(0, 4, "GPU-") == 0 || u.compare(0, 4, "MIG-") == 0)
		u = u.substr(4);
	int hexDigits = 0;
	for (size_t i = 0; i < u.size(); i++)
	{
		if (std::isxdigit((unsigned char)u[i]))
			hexDigits++;
		else if (u[i] != '-')
			return false;
	}
	return hexDigits == 32;
}

std::vector<std::string> EnvironmentDeviceIdSource::splitList(const std::string &text)
{
	std::vector<std::string> out;
	std::stringstream ss(text);
	std::string item;
	while (std::getline(ss, item, ','))
	{
		const size_t first = item.find_first_not_of(" \t");
		const size_t last = item.find_last_not_of(" \t");
		out.push_back(first == std::string::npos ? std::string() : item.substr(first, last - first + 1));
	}
	return out;
}

// ------------------------------------------------------------------ IndexDeviceIdSource

std::string IndexDeviceIdSource::identify(int deviceIndex) const
{
	std::stringstream s;
	s << hostName << "/index:" << deviceIndex;
	return s.str();
}

// ------------------------------------------------------------------ DeviceIdResolver

DeviceIdResolver DeviceIdResolver::forThisBackend(int deviceCount, const std::string &hostName)
{
	DeviceIdResolver resolver;
	resolver.fallbackHostName = hostName;

#ifdef _CUDA_ENABLED
	resolver.add(std::unique_ptr<DeviceIdSource>(new CudaUuidDeviceIdSource()));
	resolver.add(std::unique_ptr<DeviceIdSource>(new EnvironmentDeviceIdSource(deviceCount, "CUDA_VISIBLE_DEVICES")));
	resolver.add(std::unique_ptr<DeviceIdSource>(new CudaPciDeviceIdSource(hostName)));
#elif _HIP_ENABLED
	resolver.add(std::unique_ptr<DeviceIdSource>(new HipUuidDeviceIdSource()));
	resolver.add(std::unique_ptr<DeviceIdSource>(new EnvironmentDeviceIdSource(deviceCount, "HIP_VISIBLE_DEVICES")));
	resolver.add(std::unique_ptr<DeviceIdSource>(new HipPciDeviceIdSource(hostName)));
#else
	resolver.add(std::unique_ptr<DeviceIdSource>(new EnvironmentDeviceIdSource(deviceCount, "CUDA_VISIBLE_DEVICES")));
#endif
	resolver.add(std::unique_ptr<DeviceIdSource>(new IndexDeviceIdSource(hostName)));

	const char *preferred = getenv("RELION_GPU_UID_SOURCE");
	if (preferred != NULL)
		resolver.preferSource(preferred);
	return resolver;
}

void DeviceIdResolver::add(std::unique_ptr<DeviceIdSource> source)
{
	sources.push_back(std::move(source));
}

void DeviceIdResolver::preferSource(const std::string &sourceName)
{
	for (size_t i = 0; i < sources.size(); i++)
		if (sources[i]->name() == sourceName)
		{
			std::unique_ptr<DeviceIdSource> chosen = std::move(sources[i]);
			sources.erase(sources.begin() + i);
			sources.insert(sources.begin(), std::move(chosen));
			return;
		}
}

std::string DeviceIdResolver::identify(int deviceIndex) const
{
	for (size_t i = 0; i < sources.size(); i++)
	{
		const std::string id = sources[i]->identify(deviceIndex);
		if (!id.empty())
			return id;
	}
	return IndexDeviceIdSource(fallbackHostName).identify(deviceIndex);
}
