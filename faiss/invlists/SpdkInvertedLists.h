/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_SPDK_INVERTED_LISTS_H
#define FAISS_SPDK_INVERTED_LISTS_H

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <vector>

#include <faiss/invlists/InvertedLists.h>

// Forward declarations — callers do not need to include SPDK headers directly.
struct spdk_nvme_ctrlr;
struct spdk_nvme_ns;
struct spdk_nvme_qpair;

namespace faiss {

/**
 * SpdkInvertedLists: stores Faiss inverted lists on a raw NVMe block device
 * accessed via the SPDK userspace NVMe driver.
 *
 * Unlike OnDiskInvertedLists (which uses mmap), this class bypasses the OS
 * kernel entirely for data I/O.  Every call to get_codes() / get_ids() issues
 * a synchronous NVMe read through SPDK and returns a heap-allocated buffer;
 * the caller must release it via release_codes() / release_ids().
 *
 * On-device data layout per list (identical to OnDiskInvertedLists):
 *
 *   [ uint8_t codes[capacity * code_size] ][ idx_t ids[capacity] ]
 *
 * List metadata (sizes, offsets) is persisted to a small binary file
 * (metadata_path) so the index survives process restarts.
 *
 * Usage requirements:
 *   - DPDK hugepages must be configured before constructing this object.
 *   - The process typically needs elevated privileges (root or CAP_SYS_ADMIN)
 *     to bind the NVMe device to the VFIO/UIO driver used by SPDK.
 *   - Only one SpdkInvertedLists instance per NVMe namespace is supported
 *     (SPDK takes exclusive ownership of the device).
 */
struct SpdkInvertedLists : InvertedLists {
    // ------------------------------------------------------------------ //
    //  Per-list metadata                                                  //
    // ------------------------------------------------------------------ //

    struct List {
        size_t size;     ///< number of live entries
        size_t capacity; ///< allocated capacity (entries)
        uint64_t offset; ///< byte offset of this list on the NVMe device;
                         ///< UINT64_MAX means "not yet allocated"
        List();
    };

    /// One entry per inverted list (size == nlist).
    std::vector<List> lists;

    // ------------------------------------------------------------------ //
    //  Free-space management (same strategy as OnDiskInvertedLists)       //
    // ------------------------------------------------------------------ //

    struct Slot {
        uint64_t offset;   ///< byte offset on device
        uint64_t capacity; ///< byte capacity
        Slot(uint64_t offset, uint64_t capacity);
        Slot();
    };

    /// Sorted free-slot list (by offset).
    std::list<Slot> slots;

    // ------------------------------------------------------------------ //
    //  Metadata persistence                                               //
    // ------------------------------------------------------------------ //

    std::string metadata_path; ///< path to the binary metadata file
    uint64_t totsize;          ///< total bytes claimed on the NVMe device

    // ------------------------------------------------------------------ //
    //  SPDK handles                                                       //
    // ------------------------------------------------------------------ //

    struct spdk_nvme_ctrlr* ctrlr; ///< NVMe controller handle
    struct spdk_nvme_ns* ns;       ///< NVMe namespace handle (namespace 1)
    struct spdk_nvme_qpair* qpair; ///< I/O queue pair
    uint32_t sector_size;          ///< logical block size in bytes
    bool read_only;

    // ------------------------------------------------------------------ //
    //  Construction / destruction                                         //
    // ------------------------------------------------------------------ //

    /**
     * Open (or create) an SpdkInvertedLists backed by a raw NVMe device.
     *
     * The transport is selected by the format of @p trid_or_pcie:
     *
     *  PCIe (real SSD):
     *    Pass the BDF string directly, e.g. "0000:01:00.0".
     *    Pass nullptr to attach the first NVMe device found by probe.
     *
     *  NVMe-over-Fabrics TCP (recommended for emulation / testing):
     *    Pass a full trid string, e.g.:
     *      "trtype:TCP adrfam:IPv4 traddr:127.0.0.1 trsvcid:4420
     *       subnqn:nqn.2024-01.io.spdk:faiss"
     *    The target can be an SPDK nvmf_tgt backed by a malloc bdev —
     *    the device then lives entirely in RAM and no real SSD is touched.
     *    See scripts/nvmf_malloc_target.sh to start such a target.
     *
     * @param nlist              number of inverted lists
     * @param code_size          bytes per compressed vector code
     * @param trid_or_pcie       transport identifier (see above)
     * @param metadata_path      path of the binary metadata file
     *                           (created if absent)
     */
    SpdkInvertedLists(
            size_t nlist,
            size_t code_size,
            const char* trid_or_pcie,
            const char* metadata_path);

    ~SpdkInvertedLists() override;

    // ------------------------------------------------------------------ //
    //  InvertedLists read interface                                       //
    // ------------------------------------------------------------------ //

    size_t list_size(size_t list_no) const override;

    /**
     * Reads codes from NVMe and returns a heap-allocated buffer.
     * MUST be released with release_codes() when done.
     */
    const uint8_t* get_codes(size_t list_no) const override;

    /**
     * Reads IDs from NVMe and returns a heap-allocated buffer.
     * MUST be released with release_ids() when done.
     */
    const idx_t* get_ids(size_t list_no) const override;

    /// Frees the buffer returned by get_codes().
    void release_codes(size_t list_no, const uint8_t* codes) const override;

    /// Frees the buffer returned by get_ids().
    void release_ids(size_t list_no, const idx_t* ids) const override;

    // ------------------------------------------------------------------ //
    //  InvertedLists write interface                                      //
    // ------------------------------------------------------------------ //

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;

    // ------------------------------------------------------------------ //
    //  Metadata persistence (public so callers can checkpoint manually)   //
    // ------------------------------------------------------------------ //

    void save_metadata() const;
    void load_metadata();

    // ------------------------------------------------------------------ //
    //  Internals                                                          //
    // ------------------------------------------------------------------ //

private:
    /// Serialises all qpair accesses (SPDK qpairs are single-threaded).
    mutable std::mutex io_mutex;

    /// Detect whether trid_or_pcie is a full NVMf trid ("trtype:…") or a
    /// bare PCIe BDF / nullptr and dispatch to the appropriate init path.
    void init_spdk(const char* trid_or_pcie);

    /// Connect to a specific NVMf target (TCP / RDMA) using a trid string.
    /// Used for emulated targets (malloc bdev) and remote NVMe devices.
    void init_spdk_fabrics(const char* trid_str);

    /// Probe the PCIe bus and attach the first matching controller.
    /// Used when a bare BDF ("0000:01:00.0") or nullptr is supplied.
    void init_spdk_pcie(const char* pcie_addr);

    void cleanup_spdk();

    /// Synchronous NVMe read: copy [byte_offset, byte_offset+size) → buf.
    /// buf does NOT need to be DMA-aligned.
    void nvme_read(uint64_t byte_offset, size_t size, void* buf) const;

    /// Synchronous NVMe write: flush buf[0..size) → device at byte_offset.
    /// If the write range is not sector-aligned a read-modify-write is used.
    /// buf does NOT need to be DMA-aligned.
    void nvme_write(uint64_t byte_offset, size_t size, const void* buf);

    /// Allocate a contiguous byte range on the device; grow totsize if needed.
    uint64_t allocate_slot(uint64_t capacity_bytes);

    /// Return a byte range to the free list.
    void free_slot(uint64_t offset, uint64_t capacity_bytes);

    /// Resize a list while io_mutex is already held by the caller.
    void resize_locked(size_t list_no, size_t new_size);

    /// Write entries to device while io_mutex is already held by the caller.
    void update_entries_locked(
            size_t list_no,
            size_t entry_offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* codes);
};

} // namespace faiss

#endif // FAISS_SPDK_INVERTED_LISTS_H
