#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

import os
import sys
import subprocess

def showHelp():
    print(f"""
    Usage:
        {sys.argv[0]} build <build_type> <build_dir> <install_dir> [<build_options>...]
        {sys.argv[0]} (clean | clean_externals) <build_dir>
    
    Example:
        {sys.argv[0]} build release build-release product-rel
    
    Options:
        build_type       Valid build types: release, debug
        build_dir        CMake build directory name.
        install_dir      Product installation directory name.
        build_options    CMake build options.
                         i.e -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_MAKE_PROGRAM=ninja -G Ninja
    """)

def checkBuildType(buildType):
    return buildType in ("release", "debug")

def runShellCmd(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except:
        pass

def build():
    # Check if there are at least five parameters.
    if len(sys.argv) < 5:
        showHelp()
        sys.exit(1)

    buildType = sys.argv[2]
    buildDir = sys.argv[3]
    installDir = sys.argv[4]
    buildOptions = sys.argv[5:]

    # Check if the build type is allowed.
    if not checkBuildType(buildType):
        showHelp()
        sys.exit(1)

    # Run cmake commands.
    cmakeCmd = [
        f"cmake -S . -B {buildDir} -DCMAKE_BUILD_TYPE={buildType} -DCMAKE_INSTALL_PREFIX={installDir}",
        "-DAIX_BUILD_EXAMPLES=ON -DAIX_BUILD_TESTS=ON -DAIX_BUILD_STATIC=ON"
    ]
    cmakeCmd.extend(buildOptions)
    runShellCmd(" ".join(cmakeCmd))
    runShellCmd(f"cmake --build {buildDir} --parallel {os.cpu_count()} --target install")

def clean(target):
    # Check if there are at least three parameters.
    if len(sys.argv) < 3:
        showHelp()
        sys.exit(1)

    buildDir = sys.argv[2]
    runShellCmd(f"cmake --build {buildDir} --target {target}")

def main():
    if len(sys.argv) < 2:
        showHelp()
        sys.exit(1)

    if sys.argv[1] == 'build':
        build()
    elif sys.argv[1] in ['clean', 'clean_externals']:
        clean(target=sys.argv[1])
    else:
        showHelp()

if __name__ == "__main__":
    main()
