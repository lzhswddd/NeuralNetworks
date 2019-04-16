@echo off
if exist "CMakeFiles" (
    rmdir /S /Q "CMakeFiles"
    if exist "CMakeFiles" (
        echo delete "CMakeFiles" fail! 
    ) else (
        echo delete "CMakeFiles" success!
    )
)
if exist "Debug" (
    rmdir /S /Q "Debug"
    if exist "Debug" (
        echo delete "Debug" fail! 
    ) else (
        echo delete "Debug" success!
    )
)
if exist "Release" (
    rmdir /S /Q "Release"
    if exist "Release" (
        echo delete "Release" fail! 
    ) else (
        echo delete "Release" success!
    )
)
if exist "MinSizeRel" (
    rmdir /S /Q "MinSizeRel"
    if exist "MinSizeRel" (
        echo delete "MinSizeRel" fail! 
    ) else (
        echo delete "MinSizeRel" success!
    )
)
if exist "RelWithDebInfo" (
    rmdir /S /Q "RelWithDebInfo"
    if exist "RelWithDebInfo" (
        echo delete "RelWithDebInfo" fail! 
    ) else (
        echo delete "RelWithDebInfo" success!
    )
)
if exist "Win32" (
    rmdir /S /Q "Win32"
    if exist "Win32" (
        echo delete "Win32" fail! 
    ) else (
        echo delete "Win32" success!
    )
)
if exist "x64" (
    rmdir /S /Q "x64"
    if exist "x64" (
        echo delete "x64" fail! 
    ) else (
        echo delete "x64" success!
    )
)
if exist ".vs" (
    rmdir /S /Q ".vs"
    if exist ".vs" (
        echo delete ".vs" fail! 
    ) else (
        echo delete ".vs" success!
    )
)
if exist "Net.dir" (
    rmdir /S /Q "Net.dir"
    if exist "Net.dir" (
        echo delete "Net.dir" fail! 
    ) else (
        echo delete "Net.dir" success!
    )
)
if exist "ALL_BUILD.vcxproj" (
    del /A /F /Q "ALL_BUILD.vcxproj"
    if exist "ALL_BUILD.vcxproj" (
        echo delete "ALL_BUILD.vcxproj" fail! 
    ) else (
        echo delete "ALL_BUILD.vcxproj" success!
    )
)
if exist "ALL_BUILD.vcxproj.filters" (
    del /A /F /Q "ALL_BUILD.vcxproj.filters"
    if exist "ALL_BUILD.vcxproj.filters" (
        echo delete "ALL_BUILD.vcxproj.filters" fail! 
    ) else (
        echo delete "ALL_BUILD.vcxproj.filters" success!
    )
)
if exist "ALL_BUILD.vcxproj.user" (
    del /A /F /Q "ALL_BUILD.vcxproj.user"
    if exist "ALL_BUILD.vcxproj.user" (
        echo delete "ALL_BUILD.vcxproj.user" fail! 
    ) else (
        echo delete "ALL_BUILD.vcxproj.user" success!
    )
)
if exist "cmake_install.cmake" (
    del /A /F /Q "cmake_install.cmake"
    if exist "cmake_install.cmake" (
        echo delete "cmake_install.cmake" fail! 
    ) else (
        echo delete "cmake_install.cmake" success!
    )
)
if exist "CMakeCache.txt" (
    del /A /F /Q "CMakeCache.txt"
    if exist "CMakeCache.txt" (
        echo delete "CMakeCache.txt" fail! 
    ) else (
        echo delete "CMakeCache.txt" success!
    )
)
if exist "Net.sln" (
    del /A /F /Q "Net.sln"
    if exist "Net.sln" (
        echo delete "Net.sln" fail! 
    ) else (
        echo delete "Net.sln" success!
    )
)
if exist "Net.vcxproj" (
    del /A /F /Q "Net.vcxproj"
    if exist "Net.vcxproj" (
        echo delete "Net.vcxproj" fail! 
    ) else (
        echo delete "Net.vcxproj" success!
    )
)
if exist "Net.vcxproj.filters" (
    del /A /F /Q "Net.vcxproj.filters"
    if exist "Net.vcxproj.filters" (
        echo delete "Net.vcxproj.filters" fail! 
    ) else (
        echo delete "Net.vcxproj.filters" success!
    )
)
if exist "Net.vcxproj.user" (
    del /A /F /Q "Net.vcxproj.user"
    if exist "Net.vcxproj.user" (
        echo delete "Net.vcxproj.user" fail! 
    ) else (
        echo delete "Net.vcxproj.user" success!
    )
)
if exist "ZERO_CHECK.vcxproj" (
    del /A /F /Q "ZERO_CHECK.vcxproj"
    if exist "ZERO_CHECK.vcxproj" (
        echo delete "ZERO_CHECK.vcxproj" fail! 
    ) else (
        echo delete "ZERO_CHECK.vcxproj" success!
    )
)
if exist "ZERO_CHECK.vcxproj.filters" (
    del /A /F /Q "ZERO_CHECK.vcxproj.filters"
    if exist "ZERO_CHECK.vcxproj.filters" (
        echo delete "ZERO_CHECK.vcxproj.filters" fail! 
    ) else (
        echo delete "ZERO_CHECK.vcxproj.filters" success!
    )
)