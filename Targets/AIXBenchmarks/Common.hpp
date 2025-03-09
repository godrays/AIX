//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

// Project includes
#include <aix.hpp>
#include <aixDevices.hpp>
// External includes
// System includes
#include <string>
#include <unordered_set>
#include <functional>

struct AIXBenchmarkConfigs
{
    aix::DeviceType deviceType{aix::DeviceType::kCPU};
    size_t warmupCount{1};
    size_t samplingCount{1};
    size_t iterationCount{1000};
    std::string filterPattern;
    const std::unordered_set<std::string> testList;
};

struct AIXBenchmarkResult
{
    double min{INT_MAX};
    double max{-INT_MAX};
    double avg{0};
};

struct BenchmarkBase
{
public:
    virtual ~BenchmarkBase() = default;
    virtual void setup(const AIXBenchmarkConfigs& configs) = 0;
    virtual void run(const AIXBenchmarkConfigs& configs) = 0;
    virtual void cleanUp() = 0;
    std::string name() { return m_name; }
    void name(const std::string& name) { m_name = name; }
private:
    std::string m_name;
};

#define BENCHMARK(className, benchName)                 \
std::unique_ptr<BenchmarkBase> create##className()      \
{                                                       \
    auto benchmark = std::make_unique<className>();     \
    benchmark->name(benchName);                         \
    return benchmark;                                   \
}

static std::vector<std::function<std::unique_ptr<BenchmarkBase>()>> registeredBenchmarksList;
static std::unordered_map<std::string, std::function<std::unique_ptr<BenchmarkBase>()>> registeredBenchmarksMap;

#define REGISTER_BENCHMARK(className)                               \
{                                                                   \
    extern std::unique_ptr<BenchmarkBase> create##className();      \
    auto benchmark = create##className();                           \
    registeredBenchmarksMap[benchmark->name()] = create##className; \
    registeredBenchmarksList.emplace_back(create##className);       \
}
