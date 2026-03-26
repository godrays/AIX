//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Project includes
// External includes
#include <Metal/Metal.hpp>
// System includes
#include <map>
#include <mutex>
#include <set>


namespace aix::metal
{

class MTLBufferCache
{
public:
    // Constructor
    MTLBufferCache() = default;

    // Destructor
    ~MTLBufferCache()
    {
        std::lock_guard<std::mutex>  lock(m_syncObj);
        clearCache();
    }

    size_t size()
    {
        std::lock_guard<std::mutex>  lock(m_syncObj);
        return m_cacheSize;
    }

    void clear()
    {
        std::lock_guard<std::mutex>  lock(m_syncObj);
        clearCache();
    }

    MTL::Buffer* reuse(size_t size)
    {
        std::lock_guard<std::mutex>  lock(m_syncObj);
        MTL::Buffer* buffer{nullptr};

        // Find the closest buffer in the cached buffers with a similar size.
        auto it = m_cacheMap.lower_bound(size);

        while (it != m_cacheMap.end() && it->first < std::min(size << 1, size + (vm_page_size << 1)))
        {
            BufferHolder* bufHolder = it->second;
            if (bufHolder->buffer)
            {
                buffer = bufHolder->buffer;
                m_cacheSize -= buffer->length();

                // Remove from list and map, then delete BufferHolder.
                removeFromList(bufHolder);
                m_cacheMap.erase(bufHolder->mapIter);
                delete bufHolder;
                break;
            }
            else
            {
                // Remove stale entries.
                m_cacheMap.erase(it++);
            }
        }

        return buffer;
    }

    void recycle(MTL::Buffer* buffer)
    {
        if (!buffer) return;
        std::lock_guard<std::mutex> lock(m_syncObj);

        // Add to the cache.
        auto bufHolder = new BufferHolder(buffer);
        addAtHead(bufHolder);
        m_cacheSize += buffer->length();
        bufHolder->mapIter = m_cacheMap.insert({buffer->length(), bufHolder});
    }

    void reduceSize(size_t bytesToFree)
    {
        std::lock_guard<std::mutex>  lock(m_syncObj);
        if (bytesToFree < static_cast<size_t>(m_cacheSize * 0.9f))
        {
            size_t totalBytesFreed = 0;
            while (m_bhTail && (totalBytesFreed < bytesToFree))
            {
                auto bufHolder = m_bhTail;
                if (bufHolder->buffer)
                {
                    totalBytesFreed += bufHolder->buffer->length();
                    bufHolder->buffer->release();
                }

                // Remove from list and map, then delete BufferHolder.
                removeFromList(bufHolder);
                m_cacheMap.erase(bufHolder->mapIter);
                delete bufHolder;
            }
            assert(m_cacheSize >= totalBytesFreed);
            m_cacheSize -= totalBytesFreed;
        }
        else
        {
            clearCache();
        }
    }

private:
    struct BufferHolder
    {
        explicit BufferHolder(MTL::Buffer* buffer) : buffer(buffer) { }
        MTL::Buffer*  buffer{nullptr};
        BufferHolder* prev{nullptr};
        BufferHolder* next{nullptr};
        std::multimap<size_t, BufferHolder*>::iterator mapIter;
    };

    void clearCache()
    {
        for (auto& [size, cacheBufHolder] : m_cacheMap)
        {
            if (cacheBufHolder->buffer)
            {
                cacheBufHolder->buffer->release();
            }
            delete cacheBufHolder;
        }
        m_cacheMap.clear();
        m_bhHead = m_bhTail = nullptr;
        m_cacheSize = 0;
    }

    void addAtHead(BufferHolder* bufHolder)
    {
        if (!bufHolder) return;

        bufHolder->prev = nullptr;
        bufHolder->next = m_bhHead;
        if (m_bhHead)
        {
            m_bhHead->prev = bufHolder;
        }
        else
        {
            m_bhTail = bufHolder;
        }
        m_bhHead = bufHolder;
    }

    void removeFromList(BufferHolder* bufHolder)
    {
        if (!bufHolder) return;

        if (bufHolder->prev)
        {
            bufHolder->prev->next = bufHolder->next;
        }
        else
        {
            m_bhHead = bufHolder->next;
        }

        if (bufHolder->next)
        {
            bufHolder->next->prev = bufHolder->prev;
        }
        else
        {
            m_bhTail = bufHolder->prev;
        }

        bufHolder->prev = bufHolder->next = nullptr;
    }

    std::multimap<size_t, BufferHolder*> m_cacheMap;
    BufferHolder* m_bhHead{nullptr};
    BufferHolder* m_bhTail{nullptr};
    size_t m_cacheSize{0};
    std::mutex m_syncObj;
};


class MetalAllocator
{
public:
    // Constructor.
    explicit MetalAllocator(MTL::Device* device, size_t alignSize=256,
                            size_t smallHeapSize=2*1024*1024,       // 2mb
                            size_t largeHeapSize=20*1024*1024)      // 20mb
        : m_device{device}, m_alignSize{alignSize}, m_smallHeapSize{smallHeapSize}, m_largeHeapSize{largeHeapSize} {}

    // Destructor.
    virtual ~MetalAllocator()
    {
        std::lock_guard<std::mutex> lock(m_syncObj);
        // Safely release heaps.
        for (auto heap : m_smallPool) heap->release();
        for (auto heap : m_largePool) heap->release();
    }

    // Allocate memory.
    MTL::Buffer* alloc(size_t size)
    {
        std::lock_guard<std::mutex>  lock(m_syncObj);
        size = roundSize(size, m_alignSize);
        std::set<MTL::Heap*>& pool = (size < m_smallHeapSize / 2) ? m_smallPool : m_largePool;
        auto heap = findBestFitHeap(pool, size);
        if (!heap)
        {
            heap = allocNewHeap(size);
            assert(heap->usedSize() == 0);
            pool.insert(heap);
        }
        return heap->newBuffer(size, MTL::ResourceStorageModeShared);
    }

    void dealloc(MTL::Buffer* buffer)
    {
        std::lock_guard<std::mutex>  lock(m_syncObj);
        assert(buffer);
        buffer->release();
    }

    void clearEmptyHeaps()
    {
        std::lock_guard<std::mutex> lock(m_syncObj);

        // Safely remove heap from m_smallPool.
        for (auto it = m_smallPool.begin(); it != m_smallPool.end(); )
        {
            auto& heap = *it;
            if (heap->usedSize() == 0)
            {
                heap->release();
                it = m_smallPool.erase(it);     // Erase and get the next valid iterator.
            }
            else
            {
                ++it;       // Move to the next element if no erase occurs.
            }
        }

        // Safely remove heaps from m_largePool.
        for (auto it = m_largePool.begin(); it != m_largePool.end(); )
        {
            auto& heap = *it;
            if (heap->usedSize() == 0)
            {
                heap->release();
                it = m_largePool.erase(it);     // Erase and get the next valid iterator.
            }
            else
            {
                ++it;       // Move to the next element if no erase occurs.
            }
        }
    }

private:
    // Round the size to avoid fragmentation.
    static size_t roundSize(size_t size, size_t round)
    {
        return (size < round) ? round : (size + (round-1)) / round * round;
    }

    // Find the best fit heap from the pool.
    MTL::Heap* findBestFitHeap(std::set<MTL::Heap*>& pool, size_t size) const
    {
        for (auto heap : pool)
        {
            if (heap->maxAvailableSize(m_alignSize) >= size) return heap;
        }
        return nullptr;
    }

    // Allocate a new Metal heap.
    MTL::Heap* allocNewHeap(size_t size)
    {
        size_t heapSize = 0;
        if (size < m_smallHeapSize / 2)
        {
            heapSize = m_smallHeapSize;
        }
        else if (size < m_largeHeapSize / 2)
        {
            heapSize = m_largeHeapSize;
        }
        else
        {
            heapSize = roundSize(size, m_smallHeapSize);
        }

        // Create the heap descriptor.
        auto heapDesc = MTL::HeapDescriptor::alloc()->init();
        heapDesc->setSize(heapSize);
        heapDesc->setType(MTL::HeapType::HeapTypeAutomatic);
        heapDesc->setStorageMode(MTL::StorageModeShared);
        heapDesc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
        heapDesc->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);
        auto heap = m_device->newHeap(heapDesc);    // Create a heap.
        heapDesc->release();                        // Release the heap descriptor.
        return heap;
    }

    MTL::Device* m_device{nullptr};
    size_t m_alignSize{256};
    size_t m_smallHeapSize{2*1024*1024};        //  2mb
    size_t m_largeHeapSize{20*1024*1024};       // 20mb
    std::set<MTL::Heap*> m_smallPool;
    std::set<MTL::Heap*> m_largePool;
    std::mutex m_syncObj;
};


}   // namespace aix
