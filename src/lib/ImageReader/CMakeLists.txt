#
#  Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

add_library( ImageReader STATIC
  CheckerBoardImage.cpp
  Exception.h
  ImageReader.cpp
  Stopwatch.h
  TextureInfo.cpp
  include/ImageReader/CheckerBoardImage.h
  include/ImageReader/ImageReader.h
  include/ImageReader/MultiCheckerImage.h
  include/ImageReader/TextureInfo.h
  )

target_include_directories( ImageReader PUBLIC
  include
  )

find_package( OpenEXR )
if( NOT OpenEXR_FOUND )
  message( WARNING "OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in optixDemandTexture." )
endif()

find_package( ZlibStatic )
if ( NOT ZlibStatic_FOUND )
  message( WARNING "Zlib not found (see ZlibStatic_ROOT).  Will use procedural texture in optixDemandTexture." )
endif()

if( OpenEXR_FOUND AND ZlibStatic_FOUND )
  # Build our EXRImage class if OpenEXR is available, adding the necessary include path and libraries.
  target_sources( ImageReader PRIVATE
    EXRReader.cpp
    include/ImageReader/EXRReader.h )
  target_compile_definitions( ImageReader PUBLIC OPTIX_SAMPLE_USE_OPEN_EXR )
  target_link_libraries( ImageReader PUBLIC OpenEXR::IlmImf Zlib::Static )
endif()

set_property(TARGET ImageReader PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
