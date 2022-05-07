//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "Textures/TextureRequestHandler.h"
#include "DemandLoaderImpl.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "Util/NVTXProfiling.h"

#include <DemandLoading/TileIndexing.h>

namespace demandLoading {

void TextureRequestHandler::fillRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    // Try to make sure there are free tiles to handle the request
    m_loader->freeStagedTiles( deviceIndex, stream );

    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to fill the same tile (or the mip tail).
    unsigned int index = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), index);

    // Do nothing if the request has already been filled.
    if( m_loader->getPagingSystem( deviceIndex )->isResident( pageId ) )
        return;

    if( pageId == m_startPage && m_texture->isMipmapped() )
        fillMipTailRequest( deviceIndex, stream, pageId );
    else
        fillTileRequest( deviceIndex, stream, pageId );
}

void TextureRequestHandler::fillTileRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // Get the texture sampler.  This is thread safe because the sampler is invariant once it's created,
    // and tile requests never occur before the sampler is created.
    const TextureSampler& sampler = m_texture->getSampler();

    // Unpack tile index into miplevel and tile coordinates.
    const unsigned int tileIndex = pageId - m_startPage;
    unsigned int       mipLevel;
    unsigned int       tileX;
    unsigned int       tileY;
    unpackTileIndex( sampler, tileIndex, mipLevel, tileX, tileY );

    // Allocate a tile in device memory
    TilePool*     tilePool    = m_loader->getDeviceMemoryManager( deviceIndex )->getTilePool();
    TileBlockDesc tileLocator = tilePool->allocate( sizeof( TileBuffer ) );
    if( !tileLocator.isValid() )
        return;

    // Allocate a tile in pinned memory.
    PinnedItemPool<TileBuffer>* pinnedTilePool = m_loader->getPinnedMemoryManager()->getPinnedTilePool();
    TileBuffer*                 pinnedTile     = pinnedTilePool->allocate();
    if( pinnedTile == nullptr )
    {
        tilePool->freeBlock( tileLocator );
        return;
    }

    // Read the tile (from disk) into pinned tile buffer.  We use a thread-local tile buffer to amortize the
    // allocation overhead across multiple tile requests.
    const bool ok = m_texture->readTile( mipLevel, tileX, tileY, pinnedTile->data, sizeof( TileBuffer ) );
    DEMAND_ASSERT_MSG( ok, "readTile call failed" );

    // Fill the tile 
    CUmemGenericAllocationHandle handle;
    size_t                       offset;
    tilePool->getHandle( tileLocator, &handle, &offset );
    m_texture->fillTile( deviceIndex, stream, mipLevel, tileX, tileY, pinnedTile->data, sizeof( TileBuffer ), handle, offset );

    const unsigned int lruVal = 0;
    m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, lruVal, tileLocator.getData() );

    // Free the pinned memory buffer.  This doesn't immediately reclaim it: an event is recorded on
    // the stream, and the buffer isn't reused until all preceding operations are complete,
    // including the asynchronous memcpy issued by fillTile().
    pinnedTilePool->free( pinnedTile, deviceIndex, stream );
}

void TextureRequestHandler::fillMipTailRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    const size_t mipTailSize         = m_texture->getMipTailSize();
    const bool   usePinnedTileBuffer = ( mipTailSize <= sizeof( TileBuffer ) );

    // Allocate device memory for the mip tail from TilePool.
    TilePool*     tilePool  = m_loader->getDeviceMemoryManager( deviceIndex )->getTilePool();
    TileBlockDesc tileBlock = tilePool->allocate( mipTailSize );
    if( !tileBlock.isValid() )
        return;

    // Use a TileBuffer or a MipTailBuffer depending on the size of the mip tail
    PinnedItemPool<TileBuffer>*    pinnedTilePool    = m_loader->getPinnedMemoryManager()->getPinnedTilePool();
    PinnedItemPool<MipTailBuffer>* pinnedMipTailPool = m_loader->getPinnedMemoryManager()->getPinnedMipTailPool();

    TileBuffer*    pinnedTileBuffer    = usePinnedTileBuffer ? pinnedTilePool->allocate() : nullptr;
    MipTailBuffer* pinnedMipTailBuffer = !usePinnedTileBuffer ? pinnedMipTailPool->allocate() : nullptr;
    char*          pinnedData          = usePinnedTileBuffer ? pinnedTileBuffer->data : pinnedMipTailBuffer->data;
    size_t         pinnedBuffSize      = usePinnedTileBuffer ? sizeof( TileBuffer ) : sizeof( MipTailBuffer );

    // If it failed to allocate pinned memory, just return
    if( pinnedTileBuffer == nullptr && pinnedMipTailBuffer == nullptr )
    {
        tilePool->freeBlock( tileBlock );
        return;
    }

    // Read the mip tail.
    const bool ok = m_texture->readMipTail( pinnedData, pinnedBuffSize );
    DEMAND_ASSERT_MSG( ok, "readMipTail call failed" );

    CUmemGenericAllocationHandle handle;
    size_t                       offset;
    tilePool->getHandle( tileBlock, &handle, &offset );

    // Fill the mip tail.
    m_texture->fillMipTail( deviceIndex, stream, pinnedData, mipTailSize, handle, offset );

    // Add a mapping for the mip tail, which will be sent to the device in pushMappings().
    unsigned int lruVal = 0;
    m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, lruVal, tileBlock.getData() );

    // Free the pinned memory buffer.  This doesn't immediately reclaim it: an event is recorded on
    // the stream, and the buffer isn't reused until all preceding operations are complete,
    // including the asynchronous memcpy issued by fillTile().
    if( usePinnedTileBuffer )
        pinnedTilePool->free( pinnedTileBuffer, deviceIndex, stream );
    else
        pinnedMipTailPool->free( pinnedMipTailBuffer, deviceIndex, stream );
}

void TextureRequestHandler::unmapTileResource( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to fill the same tile (or the mip tail).
    unsigned int tileIndex = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), tileIndex);

    // If the page has already been remapped, don't unmap it
    PagingSystem* pagingSystem = m_loader->getPagingSystem( deviceIndex );
    if( pagingSystem->isResident( pageId ) )
        return;

    DemandTextureImpl* texture = getTexture();
    
    // Unmap the tile or mip tail
    if( tileIndex == 0 )
    {
        texture->unmapMipTail( deviceIndex, stream );
    }
    else
    {
        unsigned int mipLevel;
        unsigned int tileX;
        unsigned int tileY;
        unpackTileIndex( texture->getSampler(), tileIndex, mipLevel, tileX, tileY );
        texture->unmapTile( deviceIndex, stream, mipLevel, tileX, tileY );
    }
}


}  // namespace demandLoading
