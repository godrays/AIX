//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#if WIN32
    #define YAML_CPP_STATIC_DEFINE
#endif

// Project includes
#include "Common.hpp"
// External includes
#include <docopt/docopt.h>
#include <yaml-cpp/yaml.h>
// System includes
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>


void registerAllBenchmarks()
{
    REGISTER_BENCHMARK(BenchmarkTensorAddF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorSubF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorMulF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorDivF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorUnaryF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorSumF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorSumWithDimF32300)
    REGISTER_BENCHMARK(BenchmarkTensorMaxF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorMaxWithDimF32300)
    REGISTER_BENCHMARK(BenchmarkTensorSqrtF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorSinF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorCosF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorTanhF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorLogF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorExpF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorPowF3210M)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF3210)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF3225)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF3250)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32100)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32200)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32500)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32800)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF321000)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF322000)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF3216)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF3232)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF3264)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32128)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32256)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32512)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32768)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF321024)
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF322048)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF3210)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF3225)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF3250)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32100)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32200)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32500)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32800)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF321000)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF322000)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF3216)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF3232)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF3264)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32128)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32256)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32512)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF32768)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF321024)
    REGISTER_BENCHMARK(BenchmarkTensorTransposeF322048)
    REGISTER_BENCHMARK(BenchmarkTensorPermuteF32250)
    REGISTER_BENCHMARK(BenchmarkTensorPermuteF32256)
    REGISTER_BENCHMARK(BenchmarkTensorSliceF323K)
    REGISTER_BENCHMARK(BenchmarkTensorCatF32V)
    REGISTER_BENCHMARK(BenchmarkTensorCatF32H)
    REGISTER_BENCHMARK(BenchmarkTensorIndexSelectF322001)
    REGISTER_BENCHMARK(BenchmarkDeviceAddF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceSubF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceMulF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceDivF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceUnaryF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceSumF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceMaxF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceSqrtF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceSinF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceCosF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceTanhF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceLogF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceExpF3210M)
    REGISTER_BENCHMARK(BenchmarkDevicePowF3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF3210)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF3225)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF3250)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32100)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32200)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32500)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32800)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF321000)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF322000)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF3216)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF3232)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF3264)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32128)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32256)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32512)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32768)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF321024)
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF322048)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF3210)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF3225)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF3250)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32100)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32200)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32500)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32800)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF321000)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF322000)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF3216)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF3232)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF3264)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32128)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32256)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32512)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF32768)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF321024)
    REGISTER_BENCHMARK(BenchmarkDeviceTransposeF322048)
    REGISTER_BENCHMARK(BenchmarkDeviceFillF32F3210M)
    REGISTER_BENCHMARK(BenchmarkDeviceCopyF32F3210M)
    REGISTER_BENCHMARK(BenchmarkModelXORF321K)
    REGISTER_BENCHMARK(BenchmarkModelXORF3210)
}


bool matchWithWildcard(const std::string& str, const std::string& pattern)
{
    size_t strIndex = 0, patIndex = 0;
    size_t starIndex = pattern.size();
    size_t matchIndex = 0;

    while (strIndex < str.size())
    {
        if (patIndex < pattern.size() && (pattern[patIndex] == str[strIndex] || pattern[patIndex] == '*'))
        {
            if (pattern[patIndex] == '*')
            {
                starIndex = patIndex;
                matchIndex = strIndex;
                patIndex++;
            }
            else
            {
                strIndex++;
                patIndex++;
            }
        }
        else if (starIndex != pattern.size())
        {
            patIndex = starIndex + 1;
            matchIndex++;
            strIndex = matchIndex;
        }
        else
        {
            return false;
        }
    }

    while (patIndex < pattern.size() && pattern[patIndex] == '*')
    {
        patIndex++;
    }

    return patIndex == pattern.size();
}


size_t longestBenchmarkLength(const AIXBenchmarkConfigs& configs)
{
    size_t maxLength = 0;
    for (const auto& benchmarkCreateFunc : registeredBenchmarksList)
    {
        auto benchmarkName = benchmarkCreateFunc()->name();
        if (!configs.filterPattern.empty() && !matchWithWildcard(benchmarkName, configs.filterPattern)) continue;
        maxLength = std::max(maxLength, benchmarkName.size());
    }
    return maxLength;
}


AIXBenchmarkResult runBenchmark(std::function<std::unique_ptr<BenchmarkBase>()>& benchmarkCreateFunc,
                                const AIXBenchmarkConfigs& configs)
{
    AIXBenchmarkResult result;
    double avgDurationSum = 0;
    for (size_t i=0; i<configs.samplingCount + configs.warmupCount; ++i)
    {
        // Create
        auto benchmark = benchmarkCreateFunc();     // Create an instance of a benchmark.

        // Setup
        benchmark->setup(configs);

        // Run
        auto minDuration = std::chrono::duration<double, std::milli>(std::numeric_limits<double>::max()).count();
        auto maxDuration = std::chrono::duration<double, std::milli>(std::numeric_limits<double>::lowest()).count();
        auto sumDuration = std::chrono::duration<double, std::milli>(0).count();
        for (size_t ic=0; ic<configs.iterationCount; ++ic)
        {
            auto timeStart = std::chrono::high_resolution_clock::now();
            benchmark->run(configs);
            auto timeEnd = std::chrono::high_resolution_clock::now();

            if (i < configs.warmupCount) continue;   // Skip timing for warm-up.

            auto duration = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
            minDuration = std::min(minDuration, duration);
            maxDuration = std::max(maxDuration, duration);
            sumDuration += duration;
        }

        // CleanUp
        benchmark->cleanUp();

        if (i < configs.warmupCount) continue;   // Skip timing for warm-up.

        result.min = std::min(result.min, minDuration);
        result.max = std::max(result.max, maxDuration);
        avgDurationSum += sumDuration / static_cast<double>(configs.iterationCount);
    }

    result.avg = avgDurationSum / static_cast<double>(configs.samplingCount);

    return result;
}


void saveBenchmarks(const std::string& filename, const AIXBenchmarkConfigs& configs)
{
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "config" << YAML::BeginMap;
    out << YAML::Key << "deviceType" << YAML::Value << static_cast<size_t>(configs.deviceType);
    out << YAML::Key << "warmupCount" << YAML::Value << configs.warmupCount;
    out << YAML::Key << "samplingCount" << YAML::Value << configs.samplingCount;
    out << YAML::Key << "iterationCount" << YAML::Value << configs.iterationCount;
    out << YAML::EndMap;

    out << YAML::Key << "benchmarks" << YAML::BeginMap;
    for (auto& benchmarkCreateFunc : registeredBenchmarksList)
    {
        auto benchmarkName = benchmarkCreateFunc()->name();
        if (!configs.filterPattern.empty() && !matchWithWildcard(benchmarkName, configs.filterPattern)) continue;
        if (!configs.testList.empty() && !configs.testList.contains(benchmarkName)) continue;
        auto result = runBenchmark(benchmarkCreateFunc, configs);

        out << YAML::Key << benchmarkName << YAML::BeginMap;
        out << YAML::Key << "min" << YAML::Value << result.min;
        out << YAML::Key << "max" << YAML::Value << result.max;
        out << YAML::Key << "avg" << YAML::Value << result.avg;
        out << YAML::EndMap;

        std::cout << "[" << benchmarkName.c_str() << "] : done!" << std::endl;
    }
    out << YAML::EndMap;
    out << YAML::EndMap;

    std::ofstream file(filename);
    file << out.c_str();
    file.close();
}


void compareBenchmarks(const std::string& filename, const AIXBenchmarkConfigs& configs)
{
    size_t maxNameLength = longestBenchmarkLength(configs);
    constexpr int timeWidth = 14;
    constexpr int changeWidth = 14;

    auto config = YAML::LoadFile(filename);
    auto benchmarks = config["benchmarks"];

    // Print the header
    std::cout << std::left << std::setw(maxNameLength) << "Name"
              << std::right << std::setw(timeWidth) << "Base (ms)"
              << std::right << std::setw(timeWidth) << "New (ms)"
              << std::right << std::setw(changeWidth) << "Change (%)"
              << std::endl << std::string(maxNameLength + 2*timeWidth + changeWidth, '-') << std::endl;

    for (auto& benchmarkCreateFunc : registeredBenchmarksList)
    {
        auto benchmarkName = benchmarkCreateFunc()->name();
        if (!configs.testList.empty() && !configs.testList.contains(benchmarkName)) continue;
        auto test = benchmarks[benchmarkName];
        if (test)
        {
            auto results = runBenchmark(benchmarkCreateFunc, configs);
            auto perfChange = 100 * (test["min"].as<double>() - results.min) / test["min"].as<double>();
            std::cout << std::left << std::setw(maxNameLength) << benchmarkName
                      << std::right << std::setw(timeWidth) << std::fixed << std::setprecision(4) << test["min"].as<double>()
                      << std::right << std::setw(timeWidth) << std::fixed << std::setprecision(4) << results.min
                      << std::right << std::setw(changeWidth) << std::fixed << std::setprecision(4) << perfChange
                      << std::endl;
        }
    }
}


void listBenchmarkNames(const AIXBenchmarkConfigs& configs)
{
    for (const auto& benchmarkCreateFunc : registeredBenchmarksList)
    {
        auto benchmarkName = benchmarkCreateFunc()->name();
        if (!configs.filterPattern.empty() && !matchWithWildcard(benchmarkName, configs.filterPattern)) continue;
        std::cout << benchmarkName << std::endl;
    }
}


int main(int argc, const char* argv[])
{
    static const char USAGE[] =
    R"(
    AIXBenchmarks - Copyright (c) 2024-Present, Arkin Terli. All rights reserved.

    Usage:
        AIXBenchmarks (save|compare) --file=<name> --device=<name> [--testName=<name>... | --filter=<pattern>]
                                                                   [--wc=<number>] [--sc=<number>] [--ic=<number>]
        AIXBenchmarks list [--filter=<pattern>]

    Options:
        save                Save benchmark results to a file. No comparison.
        compare             Run benchmarks and compare results within the specified file.
        list                List test names.

        --file=<name>       YAML file that stores benchmark results.
        --device=<name>     Device Type: CPU|MCS.  [default: CPU]
        --testName=<name>   Unique test name to run. (Could be multiple)
                            (i.e. --testName=matmul1 --testName=matmul2)
        --filter=<pattern>  Filter test names with wildcards.
                            (i.e. --filter="*matmul*")
        --wc=<number>       Warm-up count.         [default: 1]
        --sc=<number>       Sampling count.        [default: 1]
        --ic=<number>       Iteration count.       [default: 1000]
)";

    std::map <std::string, docopt::value>  args;

    try
    {
        // Check cmd-line parameters.
        std::vector<std::string>  baseArgs{ argv + 1, argv + argc };
        if (baseArgs.empty() || (baseArgs[0] != "save" && baseArgs[0] != "compare" && baseArgs[0] != "list"))
        {
            std::cout << USAGE << std::endl;
            return -1;
        }

        // Parse cmd-line parameters.
        args = docopt::docopt(USAGE, {argv + 1, argv + argc}, false, "AIXBenchmarks 0.0.0");

        auto file = args["--file"] ? args["--file"].asString() : "";
        if (args["compare"].asBool() && !std::filesystem::exists(file))
        {
            std::cerr << "Could not find file: " << file << std::endl;
            return -1;
        }

        auto deviceTypeStr = args["--device"] ? args["--device"].asString() : "CPU";
        if (deviceTypeStr != "CPU" && deviceTypeStr != "MCS")
        {
            std::cerr << "Unknown device type: " << deviceTypeStr << std::endl;
            return -1;
        }
        aix::DeviceType deviceType = aix::DeviceType::kCPU;
        if (deviceTypeStr == "MCS")
            deviceType = aix::DeviceType::kGPU_METAL;

        // Check if this device is available in this platform.
        auto device = aix::createDevice(deviceType);
        if (!device)
        {
            std::cerr << "Device is not supported in this platform" << std::endl;
            return -1;
        }
        device.release();

        auto samplingCount = args["--sc"] ? args["--sc"].asLong() : 1;
        if (samplingCount < 1)
        {
            std::cerr << "Sampling count must be bigger than one." << std::endl;
            return -1;
        }

        auto iterationCount = args["--ic"] ? args["--ic"].asLong() : 1000;
        if (iterationCount < 1)
        {
            std::cerr << "Iteration count must be bigger than one." << std::endl;
            return -1;
        }

        auto warmupCount = args["--wc"] ? args["--wc"].asLong() : 1;
        if (warmupCount < 1)
        {
            std::cerr << "Warm-up count must be bigger than one." << std::endl;
            return -1;
        }

        auto filterPattern = args["--filter"] ? args["--filter"].asString() : "";
        auto testsToRun = args["--testName"] ? args["--testName"].asStringList() : std::vector<std::string>();

        AIXBenchmarkConfigs benchConfigs
        {
            .deviceType     = deviceType,
            .warmupCount    = static_cast<size_t>(warmupCount),
            .samplingCount  = static_cast<size_t>(samplingCount),
            .iterationCount = static_cast<size_t>(iterationCount),
            .filterPattern  = filterPattern,
            .testList       = std::unordered_set<std::string>(testsToRun.begin(), testsToRun.end())
        };

        registerAllBenchmarks();

        if (args["save"].asBool())
        {
            saveBenchmarks(file, benchConfigs);
        }
        else if (args["compare"].asBool())
        {
            compareBenchmarks(file, benchConfigs);
        }
        else if (args["list"].asBool())
        {
            listBenchmarkNames(benchConfigs);
        }
        else
        {
            std::cerr << "Unknown command usage." << std::endl;
            return -1;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception message: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
