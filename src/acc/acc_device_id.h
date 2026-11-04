/***************************************************************************
 *
 * Persistent, globally unique identifiers for accelerator devices.
 *
 * Why: the MPI follower set-up in MlOptimiserMpi::initialise identified a GPU as
 * "Device <runtime index> on <hostname>". When a scheduler gives every MPI rank its
 * own GPU (e.g. Slurm --gpus-per-task), every rank sees its GPU as runtime index 0,
 * so ranks on the same host look like they share one GPU. RELION then divides the
 * GPU memory by the number of "sharing" ranks and warns about a split GPU, although
 * each rank owns a different device. The reverse can also happen: one physical GPU
 * visible under different indices in different ranks is not recognised as shared.
 *
 * getAccPersistentDeviceId() returns a string that identifies the physical device
 * (or MIG instance) independently of the process's device numbering. Sources, in the
 * order tried for RELION_GPU_UID_SOURCE=auto (the default):
 *
 *   api    vendor UUID. CUDA: cuDeviceGetUuid_v2 (MIG-instance aware; the v1 call
 *          returns the parent GPU's UUID for every MIG slice of that GPU).
 *          HIP: hipDeviceGetUuid.
 *   env    UUIDs the system put into the visible-devices variable
 *          (CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES), e.g. "GPU-...,MIG-..."
 *          as set by Slurm; or an explicit list in RELION_GPU_UIDS
 *          (comma separated, in runtime device order). Used only when every entry
 *          looks like a UUID and the count matches the runtime device count.
 *   pci    "<host>/pci:<PCI bus id>". Unique per host, but not per MIG instance.
 *   index  "<host>/index:<runtime index>": the previous behaviour.
 *
 * RELION_GPU_UID_SOURCE=api|env|pci|index forces one source (falling back to the
 * next one only if it fails).
 *
 ***************************************************************************/
#ifndef ACC_DEVICE_ID_H_
#define ACC_DEVICE_ID_H_

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#ifdef _CUDA_ENABLED
	#include <cuda.h>
	#include <cuda_runtime.h>
#elif _HIP_ENABLED
	#include <hip/hip_runtime.h>
#endif

namespace acc_device_id_detail
{
	inline std::string formatUuid(const unsigned char *b, const char *prefix)
	{
		char s[48];
		snprintf(s, sizeof s, "%s%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", prefix,
		         b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
		return s;
	}

	inline std::vector<std::string> splitList(const char *list)
	{
		std::vector<std::string> out;
		if (list == NULL) return out;
		std::stringstream ss(list);
		std::string item;
		while (std::getline(ss, item, ','))
		{
			size_t a = item.find_first_not_of(" \t"), e = item.find_last_not_of(" \t");
			out.push_back(a == std::string::npos ? std::string() : item.substr(a, e - a + 1));
		}
		return out;
	}

	inline bool looksLikeUuid(const std::string &s)
	{
		// GPU-xxxxxxxx-..., MIG-xxxxxxxx-..., or a bare 36-character UUID
		std::string u = s;
		if (u.compare(0, 4, "GPU-") == 0 || u.compare(0, 4, "MIG-") == 0) u = u.substr(4);
		if (u.size() < 32) return false;
		int hex = 0;
		for (size_t i = 0; i < u.size(); i++)
		{
			if (std::isxdigit((unsigned char)u[i])) hex++;
			else if (u[i] != '-') return false;
		}
		return hex == 32;
	}

	// vendor API UUID; empty string on failure
	inline std::string fromApi(int deviceIndex)
	{
#ifdef _CUDA_ENABLED
		CUdevice dev;
		CUuuid uuid;
		if (cuInit(0) != CUDA_SUCCESS) return "";
		if (cuDeviceGet(&dev, deviceIndex) != CUDA_SUCCESS) return "";
	#if CUDA_VERSION >= 11040
		if (cuDeviceGetUuid_v2(&uuid, dev) != CUDA_SUCCESS) return "";
	#else
		if (cuDeviceGetUuid(&uuid, dev) != CUDA_SUCCESS) return "";
	#endif
		return formatUuid(reinterpret_cast<const unsigned char *>(uuid.bytes), "uuid:");
#elif _HIP_ENABLED
		hipDevice_t dev;
		hipUUID uuid;
		if (hipDeviceGet(&dev, deviceIndex) != hipSuccess) return "";
		if (hipDeviceGetUuid(&uuid, dev) != hipSuccess) return "";
		return formatUuid(reinterpret_cast<const unsigned char *>(uuid.bytes), "uuid:");
#else
		(void)deviceIndex;
		return "";
#endif
	}

	// system-provided UUID list; empty string if absent, incomplete or not UUID-shaped
	inline std::string fromEnv(int deviceIndex, int deviceCount)
	{
		const char *explicitList = getenv("RELION_GPU_UIDS");
#ifdef _HIP_ENABLED
		const char *visible = getenv("HIP_VISIBLE_DEVICES");
#else
		const char *visible = getenv("CUDA_VISIBLE_DEVICES");
#endif
		std::vector<std::string> ids = splitList(explicitList ? explicitList : visible);
		if ((int)ids.size() != deviceCount || deviceIndex < 0 || deviceIndex >= (int)ids.size()) return "";
		for (size_t i = 0; i < ids.size(); i++)
			if (!looksLikeUuid(ids[i]) && explicitList == NULL) return "";
		if (ids[deviceIndex].empty()) return "";
		return "env:" + ids[deviceIndex];
	}

	inline std::string fromPci(int deviceIndex, const std::string &hostName)
	{
		char bus[64] = {0};
#ifdef _CUDA_ENABLED
		if (cudaDeviceGetPCIBusId(bus, sizeof bus, deviceIndex) != cudaSuccess) return "";
#elif _HIP_ENABLED
		if (hipDeviceGetPCIBusId(bus, sizeof bus, deviceIndex) != hipSuccess) return "";
#else
		(void)deviceIndex;
		return "";
#endif
		return hostName + "/pci:" + bus;
	}

	inline std::string fromIndex(int deviceIndex, const std::string &hostName)
	{
		std::stringstream s;
		s << hostName << "/index:" << deviceIndex;
		return s.str();
	}
}

/// Persistent identifier of the accelerator with runtime index deviceIndex on this process.
/// hostName is only used by the host-local fallbacks (pci, index).
inline std::string getAccPersistentDeviceId(int deviceIndex, int deviceCount, const std::string &hostName)
{
	using namespace acc_device_id_detail;
	const char *src = getenv("RELION_GPU_UID_SOURCE");
	std::string mode = src ? src : "auto";

	const char *order_auto[] = {"api", "env", "pci", "index"};
	std::vector<std::string> order;
	if (mode == "auto" || mode.empty())
		order.assign(order_auto, order_auto + 4);
	else
	{
		// forced source first, then the remaining ones in the default order
		order.push_back(mode);
		for (int i = 0; i < 4; i++)
			if (mode != order_auto[i]) order.push_back(order_auto[i]);
	}

	for (size_t i = 0; i < order.size(); i++)
	{
		std::string id;
		if (order[i] == "api") id = fromApi(deviceIndex);
		else if (order[i] == "env") id = fromEnv(deviceIndex, deviceCount);
		else if (order[i] == "pci") id = fromPci(deviceIndex, hostName);
		else if (order[i] == "index") id = fromIndex(deviceIndex, hostName);
		if (!id.empty()) return id;
	}
	return fromIndex(deviceIndex, hostName);
}

#endif /* ACC_DEVICE_ID_H_ */
