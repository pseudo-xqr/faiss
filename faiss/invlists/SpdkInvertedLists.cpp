
#include <faiss/invlists/SpdkInvertedLists.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <chrono>

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
    comp->done = true;
    if (spdk_nvme_cpl_is_error(cpl)) {
        comp->error = true;
        comp->sct = cpl->status.sct;
        comp->sc = cpl->status.sc;
    }
}

// ============================================================
// Per-thread DMA staging buffer — grows on demand, never freed
// until thread exit.  Eliminates spdk_dma_malloc/free per I/O.
// ============================================================

struct DmaBuffer {
    void*  ptr = nullptr;
    size_t cap = 0;

    // Returns a DMA-capable pointer of at least `needed` bytes aligned to
    // `alignment`.  Reallocates (doubling) only when the buffer is too small.
    void* ensure(size_t needed, uint32_t alignment) {
        if (needed <= cap) {
            return ptr;
        }
        if (ptr) {
            spdk_dma_free(ptr);
        }
        size_t new_cap = cap ? cap : size_t{65536};
        while (new_cap < needed) {
            new_cap *= 2;
        }
        ptr = spdk_dma_malloc(new_cap, alignment, nullptr);
        cap = ptr ? new_cap : 0;
        return ptr;
    }

    ~DmaBuffer() {
        if (ptr) {
            spdk_dma_free(ptr);
            ptr = nullptr;
        }
    }
};

// One read buffer and one write buffer per thread; lifetime == thread lifetime.
static thread_local DmaBuffer tl_read_dma;
static thread_local DmaBuffer tl_read_dma_codes;
static thread_local DmaBuffer tl_write_dma;

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
    opts.lcore_map = "0-63";
    // Reserve enough DMA memory for all OMP threads.
    // With OMP_NUM_THREADS=64 and 3 DMA buffers (read, read_codes, write)
    // each capped at MAX_DMA_WINDOW (4 MB), peak usage is ~768 MB.
    // 1024 MB gives a comfortable margin within the available hugepage pool.
    opts.mem_size = 1024;
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
    const char* pcie_addr; 
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

//!--- Do not recommend using it ---!//
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
//!--- Do not recommend using it ---!//

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

// One qpair per (thread x controller); created lazily on first use.
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

void SpdkInvertedLists::nvme_read_zc(
    uint64_t byte_offset,
    size_t size,
    void* dma_buf
) const {
    if (size == 0) {
        return;
    }
    struct spdk_nvme_qpair* tqpair = get_thread_qpair();
    uint32_t max_io = spdk_nvme_ns_get_max_io_xfer_size(ns);
    max_io = (max_io / sector_size) * sector_size;
    if (max_io == 0) max_io = sector_size;
    // Sector-aligned envelope that covers the entire request
    uint64_t aligned_start = (byte_offset / sector_size) * (uint64_t)sector_size;
    size_t head_skip = static_cast<size_t>(byte_offset - aligned_start);
    size_t aligned_size = align_up(head_skip + size, sector_size);

    size_t num_chunks = (aligned_size + max_io - 1) / max_io;
    std::vector<IoCompletion> comps(num_chunks);

    // Sliding window: keep at most MAX_INFLIGHT commands in the qpair at
    // once to avoid exhausting the per-qpair request pool when many OMP
    // threads submit simultaneously.
    static constexpr size_t MAX_INFLIGHT = 8;
    size_t next_submit = 0;   // index of next chunk to submit
    size_t submitted_bytes = 0;
    size_t n_done = 0;

    while (n_done < num_chunks) {
        // Submit new chunks until the window is full or all are submitted.
        while (next_submit < num_chunks &&
               next_submit - n_done < MAX_INFLIGHT) {
            size_t chunk = std::min(static_cast<size_t>(max_io),
                                    aligned_size - submitted_bytes);
            uint64_t lba = (aligned_start + submitted_bytes) / sector_size;
            uint32_t lba_count = static_cast<uint32_t>(chunk / sector_size);

            int rc = spdk_nvme_ns_cmd_read(
                    ns,
                    tqpair,
                    static_cast<char*>(dma_buf) + submitted_bytes,
                    lba,
                    lba_count,
                    io_complete_cb,
                    &comps[next_submit],
                    0);
            FAISS_THROW_IF_NOT_FMT(
                    rc == 0,
                    "SpdkInvertedLists: read cmd submission failed "
                    "(rc=%d chunk=%zu/%zu)",
                    rc,
                    next_submit,
                    num_chunks);
            submitted_bytes += chunk;
            ++next_submit;
        }

        // Poll and recount completed chunks.
        spdk_nvme_qpair_process_completions(tqpair, 0);
        n_done = 0;
        for (const auto& c : comps) {
            n_done += c.done ? 1u : 0u;
        }
    }

    for (size_t i = 0; i < num_chunks; ++i) {
        FAISS_THROW_IF_NOT_FMT(
                !comps[i].error,
                "SpdkInvertedLists: NVMe read error on chunk %zu "
                "(SCT=0x%x SC=0x%x)",
                i,
                comps[i].sct,
                comps[i].sc);
    }
}

void SpdkInvertedLists::nvme_read(
        uint64_t byte_offset,
        size_t size,
        void* buf) const {
    if (size == 0) {
        return;
    }
    // auto start = std::chrono::high_resolution_clock::now();
    // Each calling thread uses its own qpair — no serialisation needed.
    struct spdk_nvme_qpair* tqpair = get_thread_qpair();

    uint32_t max_io = spdk_nvme_ns_get_max_io_xfer_size(ns);

    max_io = (max_io / sector_size) * sector_size;
    if (max_io == 0) {
        max_io = sector_size;
    }

    // Compute the sector-aligned envelope that covers the entire request.
    uint64_t aligned_start = (byte_offset / sector_size) * (uint64_t)sector_size;
    size_t head_skip = static_cast<size_t>(byte_offset - aligned_start);
    size_t aligned_size = align_up(head_skip + size, sector_size);

    // Cap DMA window so per-thread DMA usage stays bounded.  Must be a
    // multiple of max_io (itself a multiple of sector_size).  4 MB keeps
    // 64 OMP threads within the 1024 MB pool reserved at SPDK init.
    static constexpr size_t MAX_DMA_WINDOW = 4ULL * 1024 * 1024;
    size_t dma_window = std::min(aligned_size, MAX_DMA_WINDOW);
    // Round down to max_io boundary so every window is cleanly divisible.
    dma_window = (dma_window / max_io) * max_io;
    if (dma_window == 0) {
        dma_window = max_io;
    }

    void* dma_buf = tl_read_dma.ensure(dma_window, sector_size);
    FAISS_THROW_IF_NOT_MSG(
            dma_buf != nullptr,
            "SpdkInvertedLists: failed to grow thread-local read DMA buffer");

    // Iterate over the full aligned range in dma_window-sized passes.
    size_t nvme_progress = 0; // bytes consumed from [0, aligned_size)
    size_t dst_written = 0;   // bytes written to buf

    while (nvme_progress < aligned_size) {
        size_t this_pass = std::min(dma_window, aligned_size - nvme_progress);

        // Sliding-window submission: keep at most MAX_INFLIGHT in the qpair.
        static constexpr size_t MAX_INFLIGHT = 8;
        size_t num_chunks = (this_pass + max_io - 1) / max_io;
        std::vector<IoCompletion> comps(num_chunks);

        size_t next_submit = 0;
        size_t submitted_bytes = 0;
        size_t n_done = 0;

        while (n_done < num_chunks) {
            while (next_submit < num_chunks &&
                   next_submit - n_done < MAX_INFLIGHT) {
                size_t chunk = std::min(static_cast<size_t>(max_io),
                                        this_pass - submitted_bytes);
                uint64_t lba =
                        (aligned_start + nvme_progress + submitted_bytes) /
                        sector_size;
                uint32_t lba_count =
                        static_cast<uint32_t>(chunk / sector_size);

                int rc = spdk_nvme_ns_cmd_read(
                        ns,
                        tqpair,
                        static_cast<char*>(dma_buf) + submitted_bytes,
                        lba,
                        lba_count,
                        io_complete_cb,
                        &comps[next_submit],
                        0);
                FAISS_THROW_IF_NOT_FMT(
                        rc == 0,
                        "SpdkInvertedLists: read cmd submission failed "
                        "(rc=%d chunk=%zu/%zu)",
                        rc,
                        next_submit,
                        num_chunks);
                submitted_bytes += chunk;
                ++next_submit;
            }

            spdk_nvme_qpair_process_completions(tqpair, 0);
            n_done = 0;
            for (const auto& c : comps) {
                n_done += c.done ? 1u : 0u;
            }
        }

        for (size_t i = 0; i < num_chunks; ++i) {
            FAISS_THROW_IF_NOT_FMT(
                    !comps[i].error,
                    "SpdkInvertedLists: NVMe read error on chunk %zu "
                    "(SCT=0x%x SC=0x%x)",
                    i,
                    comps[i].sct,
                    comps[i].sc);
        }

        // Copy the user-relevant portion of this pass to buf.
        // Only the first pass has a head_skip; subsequent passes start at 0.
        size_t src_offset = (nvme_progress == 0) ? head_skip : 0;
        size_t copy_bytes = this_pass - src_offset;
        copy_bytes = std::min(copy_bytes, size - dst_written);
        memcpy(static_cast<char*>(buf) + dst_written,
               static_cast<const char*>(dma_buf) + src_offset,
               copy_bytes);
        dst_written += copy_bytes;
        nvme_progress += this_pass;
    }
    
    // auto end = std::chrono::high_resolution_clock::now();
    // if (buf == nullptr) buf = static_cast<const char*>(dma_buf) + head_skip;
    // else memcpy(buf, static_cast<const char*>(dma_buf) + head_skip, size);
    // auto cpy_end = std::chrono::high_resolution_clock::now();
    // double duration_ssd = std::chrono::duration<double>(end - start).count();
    // double duration_cpy = std::chrono::duration<double>(cpy_end - end).count();
    // printf("cpy takes %f %%\n", duration_cpy*100/(duration_ssd + duration_cpy));

}

void SpdkInvertedLists::nvme_write(
        uint64_t byte_offset,
        size_t size,
        const void* buf) {
    if (size == 0) {
        return;
    }

    const char* src = static_cast<const char*>(buf);
    uint64_t cur_byte_offset = byte_offset;
    size_t bytes_remaining = size;

    uint32_t max_io_bytes = spdk_nvme_ns_get_max_io_xfer_size(ns);
    // if (max_io_bytes == 0 || max_io_bytes > 131072u) {
    //     max_io_bytes = 131072u;
    // }

    // Use the thread-local write DMA buffer; grows on demand, never freed per
    // call.  Writes are serialised under rw_mutex exclusive lock so a single
    // per-thread buffer is safe.
    size_t dma_buf_size =
            std::max(static_cast<size_t>(max_io_bytes),
                     static_cast<size_t>(sector_size));
    void* dma_buf = tl_write_dma.ensure(dma_buf_size, sector_size);
    FAISS_THROW_IF_NOT_MSG(
            dma_buf != nullptr,
            "SpdkInvertedLists: failed to grow thread-local write DMA buffer");

    while (bytes_remaining > 0) {
        uint64_t lba = cur_byte_offset / sector_size;
        uint64_t offset_in_sector = cur_byte_offset % sector_size;

        size_t current_step_size =
                std::min(bytes_remaining, dma_buf_size - offset_in_sector);

        bool is_aligned_start = (offset_in_sector == 0);
        bool is_aligned_end =
                ((offset_in_sector + current_step_size) % sector_size == 0);

        uint32_t lba_count = static_cast<uint32_t>(
                (offset_in_sector + current_step_size + sector_size - 1) /
                sector_size);

        if (!is_aligned_start || !is_aligned_end) {
            // Read-Modify-Write: read existing sectors first.
            IoCompletion read_comp;
            int rc = spdk_nvme_ns_cmd_read(
                    ns,
                    qpair,
                    dma_buf,
                    lba,
                    lba_count,
                    io_complete_cb,
                    &read_comp,
                    0);
            FAISS_THROW_IF_NOT_FMT(
                    rc == 0,
                    "SpdkInvertedLists: RMW read failed (rc=%d)",
                    rc);
            while (!read_comp.done) {
                spdk_nvme_qpair_process_completions(qpair, 0);
            }
            FAISS_THROW_IF_NOT_FMT(
                    !read_comp.error,
                    "SpdkInvertedLists: NVMe read error during RMW "
                    "(SCT=0x%x SC=0x%x)",
                    read_comp.sct,
                    read_comp.sc);
        }

        // Patch in the user data.
        memcpy(static_cast<char*>(dma_buf) + offset_in_sector,
               src,
               current_step_size);

        IoCompletion write_comp;
        int rc = spdk_nvme_ns_cmd_write(
                ns,
                qpair,
                dma_buf,
                lba,
                lba_count,
                io_complete_cb,
                &write_comp,
                0);
        FAISS_THROW_IF_NOT_FMT(
                rc == 0,
                "SpdkInvertedLists: write cmd failed (rc=%d)",
                rc);
        while (!write_comp.done) {
            spdk_nvme_qpair_process_completions(qpair, 0);
        }
        FAISS_THROW_IF_NOT_FMT(
                !write_comp.error,
                "SpdkInvertedLists: NVMe write error "
                "(SCT=0x%x SC=0x%x LBA=%lu count=%u)",
                write_comp.sct,
                write_comp.sc,
                lba,
                lba_count);

        cur_byte_offset += current_step_size;
        src += current_step_size;
        bytes_remaining -= current_step_size;
    }
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
        uint64_t new_totsize = (totsize == 0) ? 65536 : totsize * 2;
        while (new_totsize - totsize < capacity_bytes) {
            new_totsize *= 2;
        }
        uint64_t ns_bytes =
                spdk_nvme_ns_get_num_sectors(ns) * (uint64_t)sector_size;
        FAISS_THROW_IF_NOT_FMT(
                new_totsize <= ns_bytes,
                "SpdkInvertedLists: NVMe namespace capacity exceeded "
                "(need %zu bytes, device has %zu bytes)",
                static_cast<size_t>(new_totsize),
                static_cast<size_t>(ns_bytes));

        // Extend last slot if it is contiguous with the current end.
        if (!slots.empty() &&
            slots.back().offset + slots.back().capacity == totsize) {
            slots.back().capacity += new_totsize - totsize;
        } else {
            slots.emplace_back(totsize, new_totsize - totsize);
        }
        totsize = new_totsize;

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
    //! Assume that nvme_read will never perform in concurrent
    //! with write, rm the mutex.
    //std::shared_lock<std::shared_mutex> lock(rw_mutex);
    const List& l = lists[list_no];
    if (l.size == 0 || l.offset == UINT64_MAX) {
        return nullptr;
    }
    size_t n_bytes = l.size * code_size;

    // Zero-copy fast path: fits within the capped DMA window.
    uint64_t aligned_start = (l.offset / sector_size) * (uint64_t)sector_size;
    size_t head_skip = static_cast<size_t>(l.offset - aligned_start);
    size_t aligned_size = align_up(head_skip + n_bytes, sector_size);

    if (aligned_size <= tl_read_dma_codes.cap ||
        tl_read_dma_codes.ensure(aligned_size, sector_size) != nullptr) {
        nvme_read_zc(l.offset, n_bytes, tl_read_dma_codes.ptr);
        return static_cast<uint8_t*>(tl_read_dma_codes.ptr) + head_skip;
    }

    // Large list: fall back to heap buffer + chunked nvme_read.
    uint8_t* buf = new uint8_t[n_bytes];
    nvme_read(l.offset, n_bytes, buf);
    return buf;
}

const idx_t* SpdkInvertedLists::get_ids(size_t list_no) const {
    //! Assume that nvme_read will never perform in concurrent
    //! with write, rm the mutex.
    //std::shared_lock<std::shared_mutex> lock(rw_mutex);
    const List& l = lists[list_no];
    if (l.size == 0 || l.offset == UINT64_MAX) {
        return nullptr;
    }
    size_t n_bytes = l.size * sizeof(idx_t);
    // IDs are stored after codes in the allocated capacity block.
    uint64_t ids_byte_offset = l.offset + l.capacity * code_size;

    // Zero-copy fast path: fits within the capped DMA window.
    uint64_t aligned_start =
            (ids_byte_offset / sector_size) * (uint64_t)sector_size;
    size_t head_skip = static_cast<size_t>(ids_byte_offset - aligned_start);
    size_t aligned_size = align_up(head_skip + n_bytes, sector_size);

    if (aligned_size <= tl_read_dma.cap ||
        tl_read_dma.ensure(aligned_size, sector_size) != nullptr) {
        nvme_read_zc(ids_byte_offset, n_bytes, tl_read_dma.ptr);
        return reinterpret_cast<const idx_t*>(
                static_cast<const char*>(tl_read_dma.ptr) + head_skip);
    }

    // Large list: fall back to heap buffer + chunked nvme_read.
    idx_t* buf = new idx_t[l.size];
    nvme_read(ids_byte_offset, n_bytes, buf);
    return buf;
}

void SpdkInvertedLists::release_codes(
        size_t list_no,
        const uint8_t* codes) const {
    // Heap path: pointer is outside the thread-local DMA buffer.
    if (codes && codes != static_cast<uint8_t*>(tl_read_dma_codes.ptr) &&
        (tl_read_dma_codes.ptr == nullptr ||
         codes < static_cast<uint8_t*>(tl_read_dma_codes.ptr) ||
         codes >= static_cast<uint8_t*>(tl_read_dma_codes.ptr) +
                         tl_read_dma_codes.cap)) {
        delete[] codes;
    }
}

void SpdkInvertedLists::release_ids(
        size_t list_no,
        const idx_t* ids) const {
    // Heap path: pointer is outside the thread-local DMA buffer.
    const char* p = reinterpret_cast<const char*>(ids);
    if (ids && (tl_read_dma.ptr == nullptr ||
                p < static_cast<char*>(tl_read_dma.ptr) ||
                p >= static_cast<char*>(tl_read_dma.ptr) + tl_read_dma.cap)) {
        delete[] ids;
    }
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

    // // Copy existing live data to the new location.
    if (l.offset != UINT64_MAX && l.size > 0) {
        size_t n = std::min(new_size, l.size);
        size_t code_bytes = n * code_size;
        size_t id_bytes = n * sizeof(idx_t);

        // Use a temporary heap buffer (nvme_read/write handle DMA internally).
        size_t tmp_bytes = std::max(code_bytes, id_bytes);
        std::vector<uint8_t> tmp(tmp_bytes);

        // Manage codes with slots
        nvme_read(l.offset, code_bytes, tmp.data());
        
        nvme_write(new_offset, code_bytes, tmp.data());
        
        // Manage id with slots
        nvme_read(l.offset + l.capacity * code_size, id_bytes, tmp.data());

        nvme_write(new_offset + new_cap * code_size, id_bytes, tmp.data());
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
