/***************************************************************************
 *
 * Persistent, globally unique identifiers for accelerator devices.
 *
 * Why: MlOptimiserMpi::initialise used to identify a follower's GPU as
 * "Device <runtime index> on <hostname>". When the scheduler gives every MPI rank
 * its own GPU (e.g. Slurm --gpus-per-task), every rank sees its GPU as index 0,
 * so ranks on one host looked like they shared a GPU: the free memory was divided
 * between them and a "split GPU" warning printed. The reverse (one physical GPU
 * under different indices in different ranks) went unnoticed.
 *
 * A DeviceIdSource knows one way to identify a device; a DeviceIdResolver asks
 * its sources in order and returns the first answer:
 *
 *   uuid   vendor UUID (CUDA: MIG-instance aware; HIP)           globally unique
 *   env    UUIDs the system lists in CUDA_VISIBLE_DEVICES /
 *          HIP_VISIBLE_DEVICES, or an explicit RELION_GPU_UIDS   as unique as the list
 *   pci    host + PCI bus id                                     unique per host, not per MIG instance
 *   index  host + runtime index (the previous behaviour)         always available
 *
 * Default order: uuid, env, pci, index. RELION_GPU_UID_SOURCE=<name> puts that
 * source first (the others stay as fall-backs).
 *
 ***************************************************************************/
#ifndef ACC_DEVICE_ID_H_
#define ACC_DEVICE_ID_H_

#include <memory>
#include <string>
#include <vector>

/// One way of identifying an accelerator device.
class DeviceIdSource
{
public:
	virtual ~DeviceIdSource() {}

	/// Short name, used in RELION_GPU_UID_SOURCE and in messages.
	virtual std::string name() const = 0;

	/// Identifier of the device with this runtime index, or "" if this source can't tell.
	virtual std::string identify(int deviceIndex) const = 0;
};

/// UUIDs provided by the system: an explicit RELION_GPU_UIDS list (any strings), or the
/// visible-devices variable when every entry in it is a UUID (e.g. Slurm's "GPU-...,MIG-...").
class EnvironmentDeviceIdSource : public DeviceIdSource
{
public:
	EnvironmentDeviceIdSource(int deviceCount, const std::string &visibleDevicesVariable);
	std::string name() const { return "env"; }
	std::string identify(int deviceIndex) const;

	static bool looksLikeUuid(const std::string &text);
	static std::vector<std::string> splitList(const std::string &text);

private:
	int deviceCount;
	std::string visibleDevicesVariable;
};

/// Host + runtime index: the previous behaviour, used when nothing better is available.
class IndexDeviceIdSource : public DeviceIdSource
{
public:
	explicit IndexDeviceIdSource(const std::string &hostName) : hostName(hostName) {}
	std::string name() const { return "index"; }
	std::string identify(int deviceIndex) const;

private:
	std::string hostName;
};

/// Asks its sources in order; the first non-empty answer wins.
class DeviceIdResolver
{
public:
	/// Sources for the compiled backend (CUDA or HIP), ordered by RELION_GPU_UID_SOURCE.
	static DeviceIdResolver forThisBackend(int deviceCount, const std::string &hostName);

	void add(std::unique_ptr<DeviceIdSource> source);
	void preferSource(const std::string &sourceName);

	/// Identifier of the device; never empty (falls back to host + index).
	std::string identify(int deviceIndex) const;

private:
	std::vector<std::unique_ptr<DeviceIdSource> > sources;
	std::string fallbackHostName;
};

#endif /* ACC_DEVICE_ID_H_ */
