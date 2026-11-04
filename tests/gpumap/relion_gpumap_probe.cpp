// relion_gpumap_probe: a stand-alone replica of RELION's MPI GPU-assignment logic
// (src/ml_optimiser_mpi.cpp, MlOptimiserMpi::initialise, "Setup GPU related resources",
//  plus the per-device memory split in MlOptimiserMpi::initialiseWorkLoad ~L1200)
//
// Purpose: test, cheaply and without RELION, how RELION's (hostname, device-index)
// identification behaves under different Slurm GPU placements (--gpus-per-task,
// --gres per node, MIG slices, srun vs mpirun), and compare it with a proposed
// UUID-based identification. Ground truth is measured empirically from device memory.
//
// Needs only an MPI C++ compiler; the CUDA *driver* (libcuda.so.1) is dlopen'ed at run time,
// so no CUDA toolkit is needed to build. RELION uses the runtime API, which sits on top of the
// same driver enumeration (CUDA_VISIBLE_DEVICES, MIG), so device visibility is identical.
//
// Build:  mpicxx -O2 -std=c++17 -o relion_gpumap_probe relion_gpumap_probe.cpp -ldl
// Run:    srun --mpi=pmix ./relion_gpumap_probe [--gpu "0:1,1"] [--j 2] [--alloc] [--comm_mb 256]
//
// Rank 0 is the RELION "leader" and, like in RELION, never touches a GPU.

#include <mpi.h>
#include <dlfcn.h>
#include <unistd.h>
#include <signal.h>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ------------------------------------------------------------------ minimal CUDA driver API
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef unsigned long long CUdeviceptr;
struct CUuuid { unsigned char bytes[16]; };

static struct
{
	bool ok = false;
	std::string err;
	CUresult (*Init)(unsigned) = nullptr;
	CUresult (*DeviceGetCount)(int *) = nullptr;
	CUresult (*DeviceGet)(CUdevice *, int) = nullptr;
	CUresult (*DeviceGetName)(char *, int, CUdevice) = nullptr;
	CUresult (*DeviceGetUuid)(CUuuid *, CUdevice) = nullptr;     // parent GPU UUID for MIG
	CUresult (*DeviceGetUuid_v2)(CUuuid *, CUdevice) = nullptr;  // MIG-instance UUID (CUDA >= 11.4)
	CUresult (*DeviceGetPCIBusId)(char *, int, CUdevice) = nullptr;
	CUresult (*DeviceGetAttribute)(int *, int, CUdevice) = nullptr;
	CUresult (*DevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
	CUresult (*CtxSetCurrent)(CUcontext) = nullptr;
	CUresult (*MemGetInfo)(size_t *, size_t *) = nullptr;
	CUresult (*MemAlloc)(CUdeviceptr *, size_t) = nullptr;
	CUresult (*MemFree)(CUdeviceptr) = nullptr;
} cu;

template <typename F> static void sym(void *h, F &f, const char *name, bool required = true)
{
	f = (F)dlsym(h, name);
	if (!f && required && cu.err.empty()) cu.err = std::string("missing symbol ") + name;
}

static void loadCuda()
{
	void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!h) { cu.err = std::string("dlopen libcuda.so.1 failed: ") + dlerror(); return; }
	sym(h, cu.Init, "cuInit");
	sym(h, cu.DeviceGetCount, "cuDeviceGetCount");
	sym(h, cu.DeviceGet, "cuDeviceGet");
	sym(h, cu.DeviceGetName, "cuDeviceGetName");
	sym(h, cu.DeviceGetUuid, "cuDeviceGetUuid");
	sym(h, cu.DeviceGetUuid_v2, "cuDeviceGetUuid_v2", false);
	sym(h, cu.DeviceGetPCIBusId, "cuDeviceGetPCIBusId");
	sym(h, cu.DeviceGetAttribute, "cuDeviceGetAttribute");
	sym(h, cu.DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
	sym(h, cu.CtxSetCurrent, "cuCtxSetCurrent");
	sym(h, cu.MemGetInfo, "cuMemGetInfo_v2");
	sym(h, cu.MemAlloc, "cuMemAlloc_v2");
	sym(h, cu.MemFree, "cuMemFree_v2");
	if (!cu.err.empty()) return;
	CUresult r = cu.Init(0);
	if (r != 0) { cu.err = "cuInit returned " + std::to_string(r); return; }
	cu.ok = true;
}

static std::string uuidStr(const CUuuid &u)
{
	char b[48];
	const unsigned char *x = u.bytes;
	snprintf(b, sizeof b, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
	         x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
	return b;
}

// ------------------------------------------------------------------ watchdog (turns hangs into reports)
static volatile const char *g_phase = "start";
static int g_rank = -1;
static int g_timeout = 120;

static void onAlarm(int)
{
	char b[256];
	int n = snprintf(b, sizeof b, "\n!!! HANG: rank %d stuck > %d s in phase '%s' -- aborting\n",
	                 g_rank, g_timeout, (const char *)g_phase);
	if (write(2, b, n) < 0) {}
	_exit(42);
}

static void phase(const char *p)
{
	g_phase = p;
	alarm(g_timeout);
}

// ------------------------------------------------------------------ small MPI string helpers
static void sendStr(const std::string &s, int to, int tag)
{
	int n = (int)s.size();
	MPI_Send(&n, 1, MPI_INT, to, tag, MPI_COMM_WORLD);
	MPI_Send(s.data(), n, MPI_CHAR, to, tag, MPI_COMM_WORLD);
}

static std::string recvStr(int from, int tag)
{
	int n;
	MPI_Recv(&n, 1, MPI_INT, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	std::string s(n, '\0');
	MPI_Recv(&s[0], n, MPI_CHAR, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	return s;
}

static std::vector<std::string> split(const std::string &s, char d)
{
	std::vector<std::string> v;
	std::stringstream ss(s);
	std::string t;
	while (std::getline(ss, t, d)) v.push_back(t);
	return v;
}

// ------------------------------------------------------------------ verbatim copy of RELION's untangleDeviceIDs (src/args.cpp)
static void untangleDeviceIDs(std::string &tangled, std::vector<std::vector<std::string>> &untangled)
{
	size_t pos = 0;
	std::string delim = ":";
	std::vector<std::string> allRankIDs;
	std::string thisRankIDs, thisThreadID;
	while ((pos = tangled.find(delim)) != std::string::npos)
	{
		thisRankIDs = tangled.substr(0, pos);
		tangled.erase(0, pos + delim.length());
		allRankIDs.push_back(thisRankIDs);
	}
	allRankIDs.push_back(tangled);
	untangled.resize(allRankIDs.size());
	for (int i = 0; i < (int)allRankIDs.size(); i++)
	{
		pos = 0;
		delim = ",";
		while ((pos = allRankIDs[i].find(delim)) != std::string::npos)
		{
			thisThreadID = allRankIDs[i].substr(0, pos);
			allRankIDs[i].erase(0, pos + delim.length());
			untangled[i].push_back(thisThreadID);
		}
		untangled[i].push_back(allRankIDs[i]);
	}
}

// ------------------------------------------------------------------ per-follower device info
struct DevInfo
{
	std::string name, uuid1, uuid2, pci;
	int ccMajor = 0, ccMinor = 0;
	size_t total = 0;
};

struct RankInfo
{
	std::string host, env, err;
	std::vector<DevInfo> devs;
	std::string visibleSet() const // sorted MIG-aware UUIDs: the proposed grouping key
	{
		std::vector<std::string> u;
		for (auto &d : devs) u.push_back(d.uuid2);
		std::sort(u.begin(), u.end());
		std::string s;
		for (auto &x : u) s += x + ";";
		return s;
	}
};

static std::string envLine()
{
	const char *keys[] = {"SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID", "CUDA_VISIBLE_DEVICES",
	                      "SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL", "OMPI_COMM_WORLD_LOCAL_RANK",
	                      "UCX_NET_DEVICES", "NCCL_IB_HCA", "UCX_TLS"};
	std::string s;
	for (auto k : keys)
	{
		const char *v = getenv(k);
		s += std::string(k) + "=" + (v ? v : "-") + " ";
	}
	return s;
}

static std::string serialize(const RankInfo &r)
{
	std::ostringstream o;
	o << r.host << "\x1f" << r.env << "\x1f" << r.err << "\x1f" << r.devs.size();
	for (auto &d : r.devs)
		o << "\x1f" << d.name << "\x1e" << d.uuid1 << "\x1e" << d.uuid2 << "\x1e" << d.pci << "\x1e"
		  << d.ccMajor << "\x1e" << d.ccMinor << "\x1e" << d.total;
	return o.str();
}

static RankInfo deserialize(const std::string &s)
{
	auto f = split(s, '\x1f');
	RankInfo r;
	r.host = f[0];
	r.env = f[1];
	r.err = f[2];
	int n = std::stoi(f[3]);
	for (int i = 0; i < n; i++)
	{
		auto g = split(f[4 + i], '\x1e');
		DevInfo d;
		d.name = g[0]; d.uuid1 = g[1]; d.uuid2 = g[2]; d.pci = g[3];
		d.ccMajor = std::stoi(g[4]); d.ccMinor = std::stoi(g[5]); d.total = std::stoull(g[6]);
		r.devs.push_back(d);
	}
	return r;
}

// ------------------------------------------------------------------ RELION's device mapping (leader side), parametrised
// groupOf[r]  : which "host" group rank r belongs to
// posOf[r]    : the index used in the auto-mapping formula. RELION uses (rank-1) raw, not the
//               position within the host -- replicated faithfully for the legacy scheme.
static std::vector<std::vector<int>> relionMapping(const std::string &gpuArg, int size, int nr_threads,
                                                   const std::vector<int> &deviceCounts,
                                                   const std::vector<int> &ranksOnGroup,
                                                   const std::vector<int> &groupOf,
                                                   const std::vector<int> &posOf, bool legacyFormula)
{
	std::vector<std::vector<int>> map(size);
	std::string gpu_ids = gpuArg; // RELION mutates gpu_ids inside untangleDeviceIDs; we do too
	std::vector<std::vector<std::string>> allThreadIDs;
	untangleDeviceIDs(gpu_ids, allThreadIDs);

	if (allThreadIDs.size() > 0 && (int)allThreadIDs.size() < (size - 1))
	{
		int N = allThreadIDs.size();
		allThreadIDs.resize(size - 1);
		for (int rank = 1; rank < (size - 1); rank++) allThreadIDs[rank] = allThreadIDs[rank % N];
	}

	for (int rank = 1; rank < size; rank++)
	{
		bool fullAutomaticMapping = true, semiAutomaticMapping = true;
		if (((int)allThreadIDs.size() < rank) || allThreadIDs[0].size() == 0 || (!std::isdigit((unsigned char)gpu_ids.c_str()[0])))
			; // automatic
		else
		{
			fullAutomaticMapping = false;
			if ((int)allThreadIDs[rank - 1].size() == nr_threads) semiAutomaticMapping = false;
		}
		for (int i = 0; i < nr_threads; i++)
		{
			int dev_id;
			if (semiAutomaticMapping)
			{
				if (fullAutomaticMapping)
				{
					int ranksOnThisHost = ranksOnGroup[groupOf[rank]];
					int p = legacyFormula ? (rank - 1) : posOf[rank];
					if (ranksOnThisHost > 1)
						dev_id = (deviceCounts[rank] * ((p % ranksOnThisHost) * nr_threads + i)) / (ranksOnThisHost * nr_threads);
					else
						dev_id = deviceCounts[rank] * i / nr_threads;
				}
				else
					dev_id = atoi(allThreadIDs[rank - 1][i % (allThreadIDs[rank - 1]).size()].c_str());
			}
			else
				dev_id = atoi(allThreadIDs[rank - 1][i].c_str());
			map[rank].push_back(dev_id);
		}
	}
	return map;
}

// unique devices per follower, in first-use order (RELION's gpuDevices / bundles)
static std::vector<int> bundles(const std::vector<int> &threadMap)
{
	std::vector<int> b;
	for (int d : threadMap)
		if (std::find(b.begin(), b.end(), d) == b.end()) b.push_back(d);
	return b;
}

// ------------------------------------------------------------------ main
int main(int argc, char **argv)
{
	std::string gpuArg;
	int nr_threads = 1;
	bool doAlloc = false;
	long commMB = 64;
	long freeGpuMB = 0; // RELION --free_gpu_memory
	const size_t GPU_THREAD_MEMORY_OVERHEAD_MB = 200;
	const size_t PROBE = 1ull << 30; // 1 GiB marker allocation for empirical sharing detection

	for (int i = 1; i < argc; i++)
	{
		std::string a = argv[i];
		auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
		if (a == "--gpu") gpuArg = next();
		else if (a == "--j") nr_threads = std::stoi(next());
		else if (a == "--alloc") doAlloc = true;
		else if (a == "--comm_mb") commMB = std::stol(next());
		else if (a == "--free_gpu_memory") freeGpuMB = std::stol(next());
		else if (a == "--timeout") g_timeout = std::stoi(next());
		else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
	}

	signal(SIGALRM, onAlarm);
	phase("MPI_Init");
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	g_rank = rank;
	if (size < 2)
	{
		if (rank == 0) fprintf(stderr, "need >= 2 ranks (1 leader + followers), like relion_*_mpi\n");
		MPI_Finalize();
		return 2;
	}
	const bool leader = (rank == 0);
	const int T_INFO = 101, T_MAP = 102, T_RES = 103;

	// ------------------------------------------------ 1. enumerate (followers), gather at leader
	phase("enumerate");
	RankInfo me;
	{
		char buf[MPI_MAX_PROCESSOR_NAME];
		int len;
		MPI_Get_processor_name(buf, &len);
		me.host = std::string(buf, len);
		me.env = envLine();
	}
	std::vector<CUcontext> ctx;
	if (!leader)
	{
		loadCuda();
		if (!cu.ok) me.err = cu.err;
		else
		{
			int n = 0;
			cu.DeviceGetCount(&n);
			for (int i = 0; i < n; i++)
			{
				CUdevice d;
				cu.DeviceGet(&d, i);
				DevInfo di;
				char nm[256] = {0}, pci[64] = {0};
				cu.DeviceGetName(nm, sizeof nm, d);
				cu.DeviceGetPCIBusId(pci, sizeof pci, d);
				CUuuid u1{}, u2{};
				cu.DeviceGetUuid(&u1, d);
				if (cu.DeviceGetUuid_v2) cu.DeviceGetUuid_v2(&u2, d); else u2 = u1;
				cu.DeviceGetAttribute(&di.ccMajor, 75, d);
				cu.DeviceGetAttribute(&di.ccMinor, 76, d);
				CUcontext c;
				cu.DevicePrimaryCtxRetain(&c, d);
				cu.CtxSetCurrent(c);
				size_t fr, tot;
				cu.MemGetInfo(&fr, &tot);
				ctx.push_back(c);
				di.name = nm; di.pci = pci; di.uuid1 = uuidStr(u1); di.uuid2 = uuidStr(u2); di.total = tot;
				me.devs.push_back(di);
			}
		}
		sendStr(serialize(me), 0, T_INFO);
	}

	std::vector<RankInfo> all(size);
	if (leader)
	{
		all[0] = me;
		for (int r = 1; r < size; r++) all[r] = deserialize(recvStr(r, T_INFO));

		printf("==== relion_gpumap_probe: %d ranks (1 leader + %d followers), --j %d, --gpu '%s'\n",
		       size, size - 1, nr_threads, gpuArg.c_str());
		printf("---- rank table (rank 0 = leader, no GPU use)\n");
		for (int r = 0; r < size; r++)
		{
			printf("rank %2d host %-14s %s\n", r, all[r].host.c_str(), all[r].env.c_str());
			if (!all[r].err.empty()) printf("        ERROR %s\n", all[r].err.c_str());
			for (size_t i = 0; i < all[r].devs.size(); i++)
			{
				auto &d = all[r].devs[i];
				printf("        dev %zu: %s cc%d.%d pci=%s mem=%.1fGB\n"
				       "               uuid(v1/parent)=GPU-%s\n"
				       "               uuid(v2/MIG)   =%s%s\n",
				       i, d.name.c_str(), d.ccMajor, d.ccMinor, d.pci.c_str(), d.total / 1e9,
				       d.uuid1.c_str(), d.uuid1 == d.uuid2 ? "GPU-" : "MIG-", d.uuid2.c_str());
			}
		}
	}

	// ------------------------------------------------ 2. leader computes legacy + proposed mappings
	// legacy:   group = hostname,            formula position = rank-1, identifier = "Device <id> on <host>"
	// proposed: group = visible-UUID set,     formula position = index within group, identifier = MIG-aware UUID
	struct Scheme { const char *name; std::vector<std::vector<int>> map; std::vector<std::vector<int>> shares; };
	std::vector<Scheme> schemes = {{"legacy", {}, {}}, {"uuid", {}, {}}};

	if (leader)
	{
		std::vector<int> devCount(size, 0);
		for (int r = 1; r < size; r++) devCount[r] = all[r].devs.size();

		for (int s = 0; s < 2; s++)
		{
			bool legacy = (s == 0);
			std::map<std::string, int> gid;
			std::vector<int> groupOf(size, -1), posOf(size, 0), ranksOnGroup;
			for (int r = 1; r < size; r++)
			{
				std::string key = legacy ? all[r].host : all[r].visibleSet();
				if (!gid.count(key)) { gid[key] = ranksOnGroup.size(); ranksOnGroup.push_back(0); }
				groupOf[r] = gid[key];
				posOf[r] = ranksOnGroup[groupOf[r]]++;
			}
			schemes[s].map = relionMapping(gpuArg, size, nr_threads, devCount, ranksOnGroup, groupOf, posOf, legacy);

			// identifiers and share counts, exactly like RELION's second exchange
			std::map<std::string, int> idCount;
			std::vector<std::vector<std::string>> ids(size);
			for (int r = 1; r < size; r++)
				for (int d : bundles(schemes[s].map[r]))
				{
					std::string id;
					if (d < 0 || d >= devCount[r]) id = "INVALID(" + std::to_string(d) + ")@" + all[r].host;
					else id = legacy ? ("Device " + std::to_string(d) + " on " + all[r].host) : all[r].devs[d].uuid2;
					ids[r].push_back(id);
					idCount[id]++;
				}
			schemes[s].shares.resize(size);
			for (int r = 1; r < size; r++)
				for (auto &id : ids[r]) schemes[s].shares[r].push_back(idCount[id]);
		}
	}

	// send both schemes' thread maps + shares to followers
	phase("distribute-mapping");
	std::vector<std::vector<int>> myMap(2), myShares(2);
	if (leader)
	{
		for (int r = 1; r < size; r++)
			for (int s = 0; s < 2; s++)
			{
				std::string m, sh;
				for (int d : schemes[s].map[r]) m += std::to_string(d) + ",";
				for (int c : schemes[s].shares[r]) sh += std::to_string(c) + ",";
				sendStr(m, r, T_MAP);
				sendStr(sh, r, T_MAP);
			}
	}
	else
	{
		for (int s = 0; s < 2; s++)
		{
			for (auto &x : split(recvStr(0, T_MAP), ',')) if (!x.empty()) myMap[s].push_back(std::stoi(x));
			for (auto &x : split(recvStr(0, T_MAP), ',')) if (!x.empty()) myShares[s].push_back(std::stoi(x));
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// ------------------------------------------------ 3. empirical ground truth, for each scheme's own device map
	// Every follower allocates PROBE on each device it would use; afterwards the drop in free memory
	// relative to a common baseline tells how many followers *physically* share that device.
	std::vector<std::string> myErr;
	auto setDev = [&](int d) -> bool {
		if (d < 0 || d >= (int)ctx.size()) return false;
		cu.CtxSetCurrent(ctx[d]);
		return true;
	};
	std::vector<std::string> truthStr(2);
	for (int s = 0; s < 2; s++)
	{
		phase(s == 0 ? "empirical-legacy" : "empirical-uuid");
		std::vector<int> bund = leader ? std::vector<int>() : bundles(myMap[s]);
		std::vector<size_t> baseline(bund.size(), 0), after(bund.size(), 0);
		std::vector<CUdeviceptr> probes(bund.size(), 0);
		if (!leader && cu.ok)
			for (size_t b = 0; b < bund.size(); b++)
				if (setDev(bund[b])) { size_t t; cu.MemGetInfo(&baseline[b], &t); }
		MPI_Barrier(MPI_COMM_WORLD);
		if (!leader && cu.ok)
			for (size_t b = 0; b < bund.size(); b++)
				if (setDev(bund[b]) && cu.MemAlloc(&probes[b], PROBE) != 0) myErr.push_back("probe alloc failed");
		MPI_Barrier(MPI_COMM_WORLD);
		if (!leader && cu.ok)
			for (size_t b = 0; b < bund.size(); b++)
				if (setDev(bund[b])) { size_t t; cu.MemGetInfo(&after[b], &t); }
		MPI_Barrier(MPI_COMM_WORLD);
		if (!leader && cu.ok)
			for (size_t b = 0; b < bund.size(); b++)
				if (probes[b]) { setDev(bund[b]); cu.MemFree(probes[b]); }
		MPI_Barrier(MPI_COMM_WORLD);
		std::ostringstream o;
		for (size_t b = 0; b < bund.size(); b++)
			o << (baseline[b] ? llround(((double)baseline[b] - (double)after[b]) / (double)PROBE) : -1) << ",";
		truthStr[s] = o.str();
	}

	// ------------------------------------------------ 4. RELION-like greedy allocation per scheme
	// free = free / shares ; alloc = free - (free_gpu_memory + 200MB * threads on device)  (ml_optimiser_mpi.cpp ~L1218)
	std::vector<std::string> allocRes(2);
	if (doAlloc)
		for (int s = 0; s < 2; s++)
		{
			phase(s == 0 ? "alloc-legacy" : "alloc-uuid");
			std::vector<int> bund = leader ? std::vector<int>() : bundles(myMap[s]);
			std::vector<CUdeviceptr> ptr(bund.size(), 0);
			std::ostringstream o;
			MPI_Barrier(MPI_COMM_WORLD);
			if (!leader && cu.ok)
				for (size_t b = 0; b < bund.size(); b++)
				{
					if (!setDev(bund[b])) { o << " dev" << bund[b] << ":INVALID"; continue; }
					int thr = std::count(myMap[s].begin(), myMap[s].end(), bund[b]);
					size_t fr, tot;
					cu.MemGetInfo(&fr, &tot);
					fr = (size_t)((double)fr / (double)myShares[s][b]);
					size_t req = (size_t)freeGpuMB * 1000000 + GPU_THREAD_MEMORY_OVERHEAD_MB * 1000000 * thr;
					size_t sz = (fr < req) ? (size_t)(fr * 0.7) : fr - req;
					CUresult r = cu.MemAlloc(&ptr[b], sz);
					char t[160];
					snprintf(t, sizeof t, " dev%d:share=%d,asked=%.1fGB(%.0f%% of total),%s", bund[b], myShares[s][b],
					         sz / 1e9, 100.0 * sz / tot, r == 0 ? "OK" : ("FAIL(cu=" + std::to_string(r) + ")").c_str());
					o << t;
				}
			MPI_Barrier(MPI_COMM_WORLD);
			for (size_t b = 0; b < bund.size(); b++) if (ptr[b]) { setDev(bund[b]); cu.MemFree(ptr[b]); }
			allocRes[s] = o.str();
		}

	// ------------------------------------------------ 5. RELION-like host-memory MPI traffic (hang check)
	// RELION only sends host buffers: leader->followers model broadcasts, follower<->follower half-set exchange.
	phase("comm-bcast");
	std::vector<char> big((size_t)commMB << 20, (char)rank);
	double t0 = MPI_Wtime();
	MPI_Bcast(big.data(), (int)big.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
	phase("comm-reduce");
	std::vector<double> red((size_t)commMB << 17, 1.0), redOut(leader ? red.size() : 1);
	MPI_Reduce(red.data(), leader ? redOut.data() : nullptr, (int)red.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	phase("comm-halfsets");
	if (size >= 3 && (rank == 1 || rank == 2))
		MPI_Sendrecv_replace(big.data(), (int)big.size(), MPI_CHAR, 3 - rank, 7, 3 - rank, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
	double commSec = MPI_Wtime() - t0;

	// ------------------------------------------------ 6. report
	phase("report");
	if (!leader)
	{
		sendStr(truthStr[0], 0, T_RES);
		sendStr(truthStr[1], 0, T_RES);
		sendStr(allocRes[0], 0, T_RES);
		sendStr(allocRes[1], 0, T_RES);
		std::string e;
		for (auto &x : myErr) e += x + "; ";
		sendStr(e, 0, T_RES);
	}
	else
	{
		int falseSplit = 0, missedShare = 0, uuidWrong = 0, mig = 0;
		printf("---- device assignment and sharing (per follower bundle)\n");
		printf("scheme rank host           threads->dev  bundle  claimedShare  TRUE(measured)  physical device\n");
		for (int r = 1; r < size; r++)
		{
			std::vector<std::string> truth[2] = {split(recvStr(r, T_RES), ','), split(recvStr(r, T_RES), ',')};
			std::string aL = recvStr(r, T_RES), aU = recvStr(r, T_RES), e = recvStr(r, T_RES);
			for (int s = 0; s < 2; s++)
			{
				auto b = bundles(schemes[s].map[r]);
				std::string tm;
				for (int d : schemes[s].map[r]) tm += std::to_string(d) + " ";
				for (size_t i = 0; i < b.size(); i++)
				{
					int claimed = schemes[s].shares[r][i];
					int t = (i < truth[s].size() && !truth[s][i].empty()) ? std::stoi(truth[s][i]) : -1;
					bool valid = b[i] >= 0 && b[i] < (int)all[r].devs.size();
					std::string phys = valid ? all[r].devs[b[i]].uuid2.substr(0, 13) + " pci " + all[r].devs[b[i]].pci : "INVALID";
					if (valid && all[r].devs[b[i]].uuid1 != all[r].devs[b[i]].uuid2) mig = 1;
					printf("%-6s %4d %-14s %-13s dev%-3d %12d  %14d  %s%s\n", schemes[s].name, r, all[r].host.c_str(),
					       tm.c_str(), b[i], claimed, t, phys.c_str(), (t > 0 && claimed != t) ? "   <-- MISMATCH" : "");
					if (s == 0 && t > 0 && claimed > t) falseSplit++;
					if (s == 0 && t > 0 && claimed < t) missedShare++;
					if (s == 1 && t > 0 && claimed != t) uuidWrong++;
				}
			}
			if (!e.empty()) printf("        errors: %s\n", e.c_str());
			if (doAlloc)
			{
				printf("        alloc legacy:%s\n", aL.c_str());
				printf("        alloc uuid  :%s\n", aU.c_str());
			}
		}
		// does the uuid scheme pick a different (better) device map than legacy anywhere?
		int remapped = 0;
		for (int r = 1; r < size; r++) if (schemes[0].map[r] != schemes[1].map[r]) remapped++;

		printf("---- verdict\n");
		printf("legacy over-counts sharing (false 'split', wastes GPU memory) : %d bundle(s)\n", falseSplit);
		printf("legacy under-counts sharing (real sharing undetected, OOM risk): %d bundle(s)\n", missedShare);
		printf("uuid scheme disagrees with measured sharing                    : %d bundle(s)\n", uuidWrong);
		printf("followers whose auto device map differs legacy vs uuid        : %d\n", remapped);
		printf("MIG instances seen (v1 parent UUID != v2 instance UUID)        : %s\n", mig ? "yes" : "no");
		printf("comm phase (bcast+reduce+halfset, %ld MB)                      : %.2f s, no hang\n", commMB, commSec);
	}
	alarm(0);
	MPI_Finalize();
	return 0;
}
