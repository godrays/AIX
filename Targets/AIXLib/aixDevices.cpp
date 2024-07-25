//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "aix.hpp"
#include "aixDevices.hpp"
#if defined(__APPLE__) && defined(__arm64__)
#include "aixDeviceMetal.hpp"
#endif
// External includes
// System includes
#include <memory>


namespace aix
{

std::unique_ptr<aix::Device> createDevice(aix::DeviceType type, size_t deviceIndex)
{
    switch (type)
    {
        case DeviceType::kGPU_METAL:
        #if defined(__APPLE__) && defined(__arm64__)
        return std::make_unique<aix::metal::DeviceMetal>(deviceIndex);
        #else
            return nullptr;
        #endif

        case DeviceType::kCPU:
            return std::make_unique<aix::Device>(deviceIndex);

        default:
            break;
    }

    return nullptr;
}

}
