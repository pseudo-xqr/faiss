
#include <faiss/invlists/SpdkInvertedLists.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <unordered_map>

#include <spdk/env.h>
#include <spdk/nvme.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

// ============================================================
// I/O completion helper
// ============================================================

namespace {

struct IoCompletion {
    bool done = false;
    bool error = false;
    uint16_t sct; // Status Code Type
    uint16_t sc;  // Status Code
};

void io_complete_cb(void* arg, const struct spdk_nvme_cpl* cpl) {
    IoCompletion* comp = static_cast<IoCompletion*>(arg);
    // comp->error = spdk_nvme_cpl_is_error(cpl);
    comp->done = true;
    if (spdk_nvme_cpl_is_error(cpl)) {
        comp->error = true;
        // Capture specific error codes
        comp->sct = cpl->status.sct;
        comp->sc = cpl->status.sc;
    }
}

// ============================================================
// SPDK environment — initialised at most once per process
// ============================================================

bool g_spdk_env_initialized = false;

void ensure_spdk_env() {
    if (g_spdk_env_initialized) {
        return;
    }
    struct spdk_env_opts opts;
    opts.opts_size = sizeof(opts);
    spdk_env_opts_init(&opts);
    opts.name = "faiss_spdk";
    opts.shm_id = -1;        // private DPDK memory; no shared-memory segment
    opts.iova_mode = "va";   // virtual-address IOVA; works without hugepage PA access
    opts.core_mask = nullptr;
    opts.lcore_map = "0-15";
    //printf("--- coremask = %s\n", opts.core_mask);
    //printf("--- lcore_map = %s\n", opts.lcore_map);
    int rc = spdk_env_init(&opts);
    FAISS_THROW_IF_NOT_FMT(rc == 0, "spdk_env_init failed (rc=%d)", rc);
    g_spdk_env_initialized = true;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < CPU_SETSIZE; i++) CPU_SET(i, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

// ============================================================
// NVMe probe callbacks
// ============================================================

struct ProbeCtx {
    const char* pcie_addr;          ///< target BDF, or nullptr = first found
    struct spdk_nvme_ctrlr* ctrlr;  ///< filled in by attach_cb
};

bool probe_cb(
        void* cb_ctx,
        const struct spdk_nvme_transport_id* trid,
        struct spdk_nvme_ctrlr_opts* /*opts*/) {
    ProbeCtx* ctx = static_cast<ProbeCtx*>(cb_ctx);
    if (ctx->ctrlr != nullptr) {
        return false; // already found one
    }
    if (ctx->pcie_addr == nullptr) {
        return true; // accept first available
    }
    return strcmp(trid->traddr, ctx->pcie_addr) == 0;
}

void attach_cb(
        void* cb_ctx,
        const struct spdk_nvme_transport_id* /*trid*/,
        struct spdk_nvme_ctrlr* ctrlr,
        const struct spdk_nvme_ctrlr_opts* /*opts*/) {
    ProbeCtx* ctx = static_cast<ProbeCtx*>(cb_ctx);
    if (ctx->ctrlr == nullptr) {
        ctx->ctrlr = ctrlr;
    }
}

} // anonymous namespace


SpdkInvertedLists::List::List()
        : size(0), capacity(0), offset(UINT64_MAX) {}

SpdkInvertedLists::Slot::Slot(uint64_t offset, uint64_t capacity)
        : offset(offset), capacity(capacity) {}

SpdkInvertedLists::Slot::Slot() : offset(0), capacity(0) {}

SpdkInvertedLists::SpdkInvertedLists(
        size_t nlist,
        size_t code_size,
        const char* trid_or_pcie,
        const char* metadata_path)
        : InvertedLists(nlist, code_size),
          metadata_path(metadata_path),
          totsize(0),
          ctrlr(nullptr),
          ns(nullptr),
          qpair(nullptr),
          sector_size(0),
          read_only(false) {
    lists.resize(nlist);
    init_spdk(trid_or_pcie);

    // Load existing metadata if available.
    FILE* f = fopen(metadata_path, "rb");
    if (f) {
        fclose(f);
        load_metadata();
    }
}

SpdkInvertedLists::~SpdkInvertedLists() {
    // Best-effort save; ignore errors at destruction time.
    try {
        save_metadata();
    } catch (...) {
    }
    cleanup_spdk();
}

static bool is_fabrics_trid(const char* s) {
    if (s == nullptr) {
        return false;
    }
    // strncasecmp is POSIX; available on Linux.
    return strncasecmp(s, "trtype:", 7) == 0;
}

void SpdkInvertedLists::init_spdk(const char* trid_or_pcie) {
    ensure_spdk_env();
    if (is_fabrics_trid(trid_or_pcie)) {
        init_spdk_fabrics(trid_or_pcie);
    } else {
        init_spdk_pcie(trid_or_pcie);
    }
}

// Shared finalisation: obtain namespace + queue pair from ctrlr.
static void finish_ctrlr_init(
        struct spdk_nvme_ctrlr* ctrlr,
        struct spdk_nvme_ns** ns_out,
        struct spdk_nvme_qpair** qpair_out,
        uint32_t* sector_size_out) {
    // Namespace 1 is the default for single-namespace devices
    // (and the one SPDK's malloc bdev exposes).
    struct spdk_nvme_ns* ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
    FAISS_THROW_IF_NOT_MSG(
            ns != nullptr,
            "SpdkInvertedLists: failed to get NVMe namespace 1");

    uint32_t sector_size = spdk_nvme_ns_get_sector_size(ns);
    FAISS_THROW_IF_NOT_FMT(
            sector_size > 0,
            "SpdkInvertedLists: invalid sector size %u",
            sector_size);

    thread_local struct spdk_nvme_qpair* qpair =
            spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, nullptr, 0);
    FAISS_THROW_IF_NOT_MSG(
            qpair != nullptr,
            "SpdkInvertedLists: failed to allocate I/O queue pair");

    *ns_out = ns;
    *qpair_out = qpair;
    *sector_size_out = sector_size;
}

void SpdkInvertedLists::init_spdk_pcie(const char* pcie_addr) {
    // Use spdk_nvme_probe to scan the PCIe bus.  probe_cb filters by BDF
    // when pcie_addr is non-null; otherwise the first device is attached.
    ProbeCtx ctx{pcie_addr, nullptr};
    int rc = spdk_nvme_probe(nullptr, &ctx, probe_cb, attach_cb, nullptr);
    FAISS_THROW_IF_NOT_FMT(rc == 0, "spdk_nvme_probe failed (rc=%d)", rc);
    FAISS_THROW_IF_NOT_MSG(
            ctx.ctrlr != nullptr,
            "SpdkInvertedLists: no NVMe controller found via PCIe probe");

    ctrlr = ctx.ctrlr;
    finish_ctrlr_init(ctrlr, &ns, &qpair, &sector_size);
}

void SpdkInvertedLists::init_spdk_fabrics(const char* trid_str) {
    // Parse the trid string ("trtype:TCP adrfam:IPv4 traddr:127.0.0.1 …")
    // into a spdk_nvme_transport_id struct, then connect directly.
    struct spdk_nvme_transport_id trid = {};
    int rc = spdk_nvme_transport_id_parse(&trid, trid_str);
    FAISS_THROW_IF_NOT_FMT(
            rc == 0,
            "SpdkInvertedLists: failed to parse trid string \"%s\" (rc=%d)",
            trid_str,
            rc);

    // spdk_nvme_connect() is the NVMf equivalent of spdk_nvme_probe():
    // it connects to a specific target and returns the controller handle.
    ctrlr = spdk_nvme_connect(&trid, nullptr, 0);
    FAISS_THROW_IF_NOT_FMT(
            ctrlr != nullptr,
            "SpdkInvertedLists: spdk_nvme_connect failed for \"%s\"",
            trid_str);

    finish_ctrlr_init(ctrlr, &ns, &qpair, &sector_size);
}

void SpdkInvertedLists::cleanup_spdk() {
    if (qpair) {
        spdk_nvme_ctrlr_free_io_qpair(qpair);
        qpair = nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(qpairs_mutex);
        for (auto* qp : thread_qpairs) {
            spdk_nvme_ctrlr_free_io_qpair(qp);
        }
        thread_qpairs.clear();
    }
    if (ctrlr) {
        spdk_nvme_detach(ctrlr);
        ctrlr = nullptr;
    }
}

// One qpair per (thread × controller); created lazily on first use.
static thread_local std::unordered_map<
        struct spdk_nvme_ctrlr*,
        struct spdk_nvme_qpair*>
        tl_qpairs;

struct spdk_nvme_qpair* SpdkInvertedLists::get_thread_qpair() const {
    auto it = tl_qpairs.find(ctrlr);
    if (it == tl_qpairs.end() || it->second == nullptr) {
        auto* qp = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, nullptr, 0);
        
	FAISS_THROW_IF_NOT_MSG(
                qp != nullptr,
                "SpdkInvertedLists: failed to alloc per-thread I/O qpair");
        tl_qpairs[ctrlr] = qp;
        std::lock_guard<std::mutex> lk(qpairs_mutex);
        thread_qpairs.push_back(qp);
        return qp;
    }
    return it->second;
}

// ============================================================
// Metadata persistence
// ============================================================

void SpdkInvertedLists::save_metadata() const {
    FILE* f = fopen(metadata_path.c_str(), "wb");
    FAISS_THROW_IF_NOT_FMT(
            f,
            "SpdkInvertedLists: cannot open metadata file '%s' for writing: %s",
            metadata_path.c_str(),
            strerror(errno));

    auto write64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, f); };

    // Header
    write64(static_cast<uint64_t>(nlist));
    write64(static_cast<uint64_t>(code_size));
    write64(totsize);

    // Per-list metadata
    for (const List& l : lists) {
        write64(static_cast<uint64_t>(l.size));
        write64(static_cast<uint64_t>(l.capacity));
        write64(l.offset);
    }

    // Free-slot list
    write64(static_cast<uint64_t>(slots.size()));
    for (const Slot& s : slots) {
        write64(s.offset);
        write64(s.capacity);
    }

    fclose(f);
}

void SpdkInvertedLists::load_metadata() {
    FILE* f = fopen(metadata_path.c_str(), "rb");
    FAISS_THROW_IF_NOT_FMT(
            f,
            "SpdkInvertedLists: cannot open metadata file '%s' for reading: %s",
            metadata_path.c_str(),
            strerror(errno));

    auto read64 = [&]() -> uint64_t {
        uint64_t v = 0;
        fread(&v, sizeof(v), 1, f);
        return v;
    };

    uint64_t saved_nlist = read64();
    uint64_t saved_code_size = read64();
    totsize = read64();

    FAISS_THROW_IF_NOT_FMT(
            saved_nlist == nlist,
            "SpdkInvertedLists: metadata nlist mismatch (saved %zu, got %zu)",
            static_cast<size_t>(saved_nlist),
            nlist);
    FAISS_THROW_IF_NOT_FMT(
            saved_code_size == code_size,
            "SpdkInvertedLists: metadata code_size mismatch "
            "(saved %zu, got %zu)",
            static_cast<size_t>(saved_code_size),
            code_size);

    for (size_t i = 0; i < nlist; i++) {
        lists[i].size = static_cast<size_t>(read64());
        lists[i].capacity = static_cast<size_t>(read64());
        lists[i].offset = read64();
    }

    uint64_t nslots = read64();
    slots.clear();
    for (uint64_t i = 0; i < nslots; i++) {
        uint64_t off = read64();
        uint64_t cap = read64();
        slots.emplace_back(off, cap);
    }

    fclose(f);
}

// ============================================================
// Low-level NVMe I/O
// ============================================================

// Rounds size up to the next multiple of sector_size.
static inline size_t
align_up(size_t size, uint32_t sector_size) {
    return ((size + sector_size - 1) / sector_size) * sector_size;
}

void SpdkInvertedLists::nvme_read(
        uint64_t byte_offset,
        size_t size,
        void* buf) const {
    if (size == 0) {
        return;
    }

    // Each calling thread uses its own qpair — no serialisation needed.
    struct spdk_nvme_qpair* tqpair = get_thread_qpair();

    uint32_t max_io = spdk_nvme_ns_get_max_io_xfer_size(ns);
    if (max_io == 0 || max_io > 131072u) {
        max_io = 131072u;
    }
    max_io = (max_io / sector_size) * sector_size;
    if (max_io == 0) {
        max_io = sector_size;
    }

    size_t done = 0;
    while (done < size) {
        uint64_t cur_offset = byte_offset + done;
        size_t cur_size = std::min(size - done, static_cast<size_t>(max_io));

        uint64_t aligned_offset =
                (cur_offset / sector_size) * (uint64_t)sector_size;
        size_t head_skip =
                static_cast<size_t>(cur_offset - aligned_offset);
        size_t aligned_size = align_up(head_skip + cur_size, sector_size);

        void* dma_buf = spdk_dma_malloc(aligned_size, sector_size, nullptr);
        FAISS_THROW_IF_NOT_MSG(
                dma_buf != nullptr,
                "SpdkInvertedLists: spdk_dma_malloc failed for read buffer");

        uint64_t lba = aligned_offset / sector_size;
        uint32_t lba_count =
                static_cast<uint32_t>(aligned_size / sector_size);

        IoCompletion comp;
        int rc = spdk_nvme_ns_cmd_read(
                ns, tqpair, dma_buf, lba, lba_count, io_complete_cb, &comp, 0);
        FAISS_THROW_IF_NOT_FMT(
                rc == 0,
                "SpdkInvertedLists: spdk_nvme_ns_cmd_read submission failed "
                "(rc=%d)",
                rc);
        while (!comp.done) {
            spdk_nvme_qpair_process_completions(tqpair, 0);
        }
        FAISS_THROW_IF_NOT_MSG(
                !comp.error, "SpdkInvertedLists: NVMe read command failed");

        memcpy(
                static_cast<char*>(buf) + done,
                static_cast<const char*>(dma_buf) + head_skip,
                cur_size);
        spdk_dma_free(dma_buf);
        done += cur_size;
    }
}

// void SpdkInvertedLists::nvme_write(
//         uint64_t byte_offset,
//         size_t size,
//         const void* buf) {
//     if (size == 0) {
//         return;
//     }

//     uint32_t max_io = spdk_nvme_ns_get_max_io_xfer_size(ns);
//     if (max_io == 0 || max_io > 131072u) {
//         max_io = 131072u;
//     }
//     max_io = (max_io / sector_size) * sector_size;
//     if (max_io == 0) {
//         max_io = sector_size;
//     }

//     size_t done = 0;
//     while (done < size) {
//         uint64_t cur_offset = byte_offset + done;
//         size_t cur_size =
//                 std::min(size - done, static_cast<size_t>(max_io));

//         uint64_t aligned_offset =
//                 (cur_offset / sector_size) * (uint64_t)sector_size;
//         size_t head_skip =
//                 static_cast<size_t>(cur_offset - aligned_offset);
//         size_t aligned_size = align_up(head_skip + cur_size, sector_size);

//         void* dma_buf = spdk_dma_malloc(aligned_size, sector_size, nullptr);
//         FAISS_THROW_IF_NOT_MSG(
//                 dma_buf != nullptr,
//                 "SpdkInvertedLists: spdk_dma_malloc failed for write buffer");

//         memset(dma_buf, 0, aligned_size);
//         memcpy(static_cast<char*>(dma_buf) + head_skip,
//                static_cast<const char*>(buf) + done,
//                cur_size);

//         uint64_t lba = aligned_offset / sector_size;
//         uint32_t lba_count =
//                 static_cast<uint32_t>(aligned_size / sector_size);

//         IoCompletion comp;
//         int rc = spdk_nvme_ns_cmd_write(
//                 ns, qpair, dma_buf, lba, lba_count, io_complete_cb, &comp, 0);
//         FAISS_THROW_IF_NOT_FMT(
//                 rc == 0,
//                 "SpdkInvertedLists: spdk_nvme_ns_cmd_write submission failed "
//                 "(rc=%d)",
//                 rc);
//         while (!comp.done) {
//             spdk_nvme_qpair_process_completions(qpair, 0);
//         }
//         FAISS_THROW_IF_NOT_MSG(
//                 !comp.error, "SpdkInvertedLists: NVMe write command failed");

//         spdk_dma_free(dma_buf);
//         done += cur_size;
//     }
// }
void SpdkInvertedLists::nvme_write(
        uint64_t byte_offset,
        size_t size,
        const void* buf) {

    if (size == 0) return;

    const char* src = static_cast<const char*>(buf);
    uint64_t cur_byte_offset = byte_offset;
    size_t bytes_remaining = size;

    // 1. Determine max I/O size and allocate a reusable DMA buffer 
    // to avoid high-frequency allocation overhead.
    uint32_t max_io_bytes = spdk_nvme_ns_get_max_io_xfer_size(ns);
    if (max_io_bytes == 0 || max_io_bytes > 131072u) max_io_bytes = 131072u;
    
    // Ensure buffer is at least one sector large
    size_t dma_buf_size = std::max((size_t)max_io_bytes, (size_t)sector_size);
    void* dma_buf = spdk_dma_malloc(dma_buf_size, sector_size, nullptr);
    FAISS_THROW_IF_NOT(dma_buf != nullptr);

    try {
        while (bytes_remaining > 0) {
            uint64_t lba = cur_byte_offset / sector_size;
            uint64_t offset_in_sector = cur_byte_offset % sector_size;
            
            // Calculate how much we can write in this step
            size_t current_step_size = std::min(bytes_remaining, dma_buf_size - offset_in_sector);
            
            // Determine if we need Read-Modify-Write (RMW)
            // We need RMW if: 
            // a) The start is not sector-aligned
            // b) The end is not sector-aligned
            bool is_aligned_start = (offset_in_sector == 0);
            bool is_aligned_end = ((offset_in_sector + current_step_size) % sector_size == 0);

            uint32_t lba_count = (offset_in_sector + current_step_size + sector_size - 1) / sector_size;

            if (!is_aligned_start || !is_aligned_end) {
                // --- READ ---
                IoCompletion read_comp;
                int rc = spdk_nvme_ns_cmd_read(
                    ns, qpair, dma_buf, lba, lba_count, io_complete_cb, &read_comp, 0);
                
                if (rc != 0) FAISS_THROW_FMT("Read failed during RMW (rc=%d)", rc);
                while (!read_comp.done) spdk_nvme_qpair_process_completions(qpair, 0);
                if (read_comp.error) 
                    FAISS_THROW_FMT("NVMe read error during RMW, SCT=0x%x, SC=0x%x", read_comp.sct, read_comp.sc);
            }

            // --- MODIFY ---
            // Copy user data into the correct position in the DMA buffer
            memcpy(static_cast<char*>(dma_buf) + offset_in_sector, src, current_step_size);

            // --- WRITE ---
            IoCompletion write_comp;
            int rc = spdk_nvme_ns_cmd_write(
                ns, qpair, dma_buf, lba, lba_count, io_complete_cb, &write_comp, 0);

            if (rc != 0) FAISS_THROW_FMT("Write failed (rc=%d)", rc);
            while (!write_comp.done) spdk_nvme_qpair_process_completions(qpair, 0);
            if (write_comp.error) {
                FAISS_THROW_FMT("NVMe write error: SCT=0x%x, SC=0x%x (LBA: %lu, Count: %u)", 
                    write_comp.sct, write_comp.sc, lba, lba_count);
            }

            // Advance pointers
            cur_byte_offset += current_step_size;
            src += current_step_size;
            bytes_remaining -= current_step_size;
        }
    } catch (...) {
        spdk_dma_free(dma_buf);
        throw;
    }

    spdk_dma_free(dma_buf);
}

// ============================================================
// Free-space management
// ============================================================

uint64_t SpdkInvertedLists::allocate_slot(uint64_t capacity_bytes) {
    // Find the first slot that fits.
    auto it = slots.begin();
    while (it != slots.end() && it->capacity < capacity_bytes) {
        ++it;
    }

    if (it == slots.end()) {
        // Grow: double the total claimed size until we have enough room.
        uint64_t new_size = (totsize == 0) ? 65536 : totsize * 2;
        while (new_size - totsize < capacity_bytes) {
            new_size *= 2;
        }
        uint64_t ns_bytes =
                spdk_nvme_ns_get_num_sectors(ns) * (uint64_t)sector_size;
        FAISS_THROW_IF_NOT_FMT(
                new_size <= ns_bytes,
                "SpdkInvertedLists: NVMe namespace capacity exceeded "
                "(need %zu bytes, device has %zu bytes)",
                static_cast<size_t>(new_size),
                static_cast<size_t>(ns_bytes));

        // Extend last slot if it is contiguous with the current end.
        if (!slots.empty() &&
            slots.back().offset + slots.back().capacity == totsize) {
            slots.back().capacity += new_size - totsize;
        } else {
            slots.emplace_back(totsize, new_size - totsize);
        }
        totsize = new_size;

        it = slots.begin();
        while (it != slots.end() && it->capacity < capacity_bytes) {
            ++it;
        }
        assert(it != slots.end());
    }

    uint64_t offset = it->offset;
    if (it->capacity == capacity_bytes) {
        slots.erase(it);
    } else {
        it->capacity -= capacity_bytes;
        it->offset += capacity_bytes;
    }
    return offset;
}

void SpdkInvertedLists::free_slot(uint64_t offset, uint64_t capacity_bytes) {
    if (capacity_bytes == 0) {
        return;
    }

    // Find insertion point (slots are sorted by offset).
    auto it = slots.begin();
    while (it != slots.end() && it->offset <= offset) {
        ++it;
    }

    // Determine neighbours.
    uint64_t end_prev = 0;
    bool has_prev = (it != slots.begin());
    if (has_prev) {
        auto prev = it;
        --prev;
        end_prev = prev->offset + prev->capacity;
    }

    uint64_t begin_next = UINT64_MAX;
    if (it != slots.end()) {
        begin_next = it->offset;
    }

    assert(!has_prev || offset >= end_prev);
    assert(offset + capacity_bytes <= begin_next);

    // Merge with previous / next slot where possible.
    if (has_prev && offset == end_prev) {
        auto prev = it;
        --prev;
        prev->capacity += capacity_bytes;
        if (offset + capacity_bytes == begin_next) {
            prev->capacity += it->capacity;
            slots.erase(it);
        }
    } else if (offset + capacity_bytes == begin_next) {
        it->offset -= capacity_bytes;
        it->capacity += capacity_bytes;
    } else {
        slots.insert(it, Slot(offset, capacity_bytes));
    }
}

// ============================================================
// InvertedLists interface — reads
// ============================================================

size_t SpdkInvertedLists::list_size(size_t list_no) const {
    return lists[list_no].size;
}

const uint8_t* SpdkInvertedLists::get_codes(size_t list_no) const {
    //std::shared_lock<std::shared_mutex> lock(rw_mutex);
    const List& l = lists[list_no];
    if (l.size == 0 || l.offset == UINT64_MAX) {
        return nullptr;
    }
    size_t n_bytes = l.size * code_size;
    uint8_t* buf = new uint8_t[n_bytes];
    nvme_read(l.offset, n_bytes, buf);
    return buf;
}

const idx_t* SpdkInvertedLists::get_ids(size_t list_no) const {
    //std::shared_lock<std::shared_mutex> lock(rw_mutex);
    const List& l = lists[list_no];
    if (l.size == 0 || l.offset == UINT64_MAX) {
        return nullptr;
    }
    size_t n_bytes = l.size * sizeof(idx_t);
    // IDs are stored after codes in the allocated capacity block.
    uint64_t ids_byte_offset = l.offset + l.capacity * code_size;
    idx_t* buf = new idx_t[l.size];
    nvme_read(ids_byte_offset, n_bytes, buf);
    return buf;
}

void SpdkInvertedLists::release_codes(
        size_t /*list_no*/,
        const uint8_t* codes) const {
    delete[] codes;
}

void SpdkInvertedLists::release_ids(
        size_t /*list_no*/,
        const idx_t* ids) const {
    delete[] ids;
}

// ============================================================
// InvertedLists interface — writes
// ============================================================

void SpdkInvertedLists::update_entries_locked(
        size_t list_no,
        size_t entry_offset,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* codes) {
    const List& l = lists[list_no];
    assert(entry_offset + n_entry <= l.size);
    // Write codes.
    nvme_write(
            l.offset + entry_offset * code_size,
            n_entry * code_size,
            codes);

    // Write IDs (stored after the full code block).
    nvme_write(
            l.offset + l.capacity * code_size +
                    entry_offset * sizeof(idx_t),
            n_entry * sizeof(idx_t),
            ids);
}

void SpdkInvertedLists::resize_locked(size_t list_no, size_t new_size) {
    List& l = lists[list_no];

    // Fast path: new size fits within existing capacity (with hysteresis).
    if (new_size <= l.capacity && new_size > l.capacity / 2) {
        l.size = new_size;
        return;
    }

    // Release the current device slot.
    if (l.offset != UINT64_MAX) {
        free_slot(l.offset, l.capacity * (sizeof(idx_t) + code_size));
    }

    if (new_size == 0) {
        l = List();
        return;
    }

    // Round up capacity to the next power of two (mirrors OnDiskInvertedLists).
    size_t new_cap = 1;
    while (new_cap < new_size) {
        new_cap *= 2;
    }

    uint64_t new_offset =
            allocate_slot(new_cap * (sizeof(idx_t) + code_size));

    // Copy existing live data to the new location.
    if (l.offset != UINT64_MAX && l.size > 0) {
        size_t n = std::min(new_size, l.size);
        size_t code_bytes = n * code_size;
        size_t id_bytes = n * sizeof(idx_t);

        // Use a temporary heap buffer (nvme_read/write handle DMA internally).
        size_t tmp_bytes = std::max(code_bytes, id_bytes);
        std::vector<uint8_t> tmp(tmp_bytes);

        nvme_read(l.offset, code_bytes, tmp.data());
        
        nvme_write(new_offset, code_bytes, tmp.data());

        nvme_read(
                l.offset + l.capacity * code_size, id_bytes, tmp.data());

        nvme_write(
                new_offset + new_cap * code_size, id_bytes, tmp.data());
    }

    l.size = new_size;
    l.capacity = new_cap;
    l.offset = new_offset;
}

void SpdkInvertedLists::resize(size_t list_no, size_t new_size) {
    FAISS_THROW_IF_NOT(!read_only);
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    resize_locked(list_no, new_size);
}

size_t SpdkInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) {
    FAISS_THROW_IF_NOT(!read_only);
    if (n_entry == 0) {
        return 0;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    size_t old_size = lists[list_no].size;
    resize_locked(list_no, old_size + n_entry);
    update_entries_locked(list_no, old_size, n_entry, ids, code);
    return old_size;
}

void SpdkInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) {
    FAISS_THROW_IF_NOT(!read_only);
    if (n_entry == 0) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    update_entries_locked(list_no, offset, n_entry, ids, code);
}

} // namespace faiss
