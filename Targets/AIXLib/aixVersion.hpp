//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

// System includes.
#include <string>

namespace aix
{
    // Version information.
    constexpr int kVersionMajor = 0;
    constexpr int kVersionMinor = 1;
    constexpr int kVersionPatch = 0;
    
    // Version as a string (e.g., "0.1.0").
    inline std::string getVersionString()
    {
        return std::to_string(kVersionMajor) + "." + 
               std::to_string(kVersionMinor) + "." + 
               std::to_string(kVersionPatch);
    }
    
    // Get version components.
    inline void getVersion(int& major, int& minor, int& patch)
    {
        major = kVersionMajor;
        minor = kVersionMinor;
        patch = kVersionPatch;
    }
}   // aix
