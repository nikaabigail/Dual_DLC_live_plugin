/*
    ------------------------------------------------------------------

    This file is part of the Open Ephys GUI
    Copyright (C) 2026

    ------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

*/

#include "DualDLCLiveBridge.h"

#include <PluginInfo.h>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#define EXPORT __declspec (dllexport)
#else
#define EXPORT __attribute__ ((visibility ("default")))
#endif

using namespace Plugin;
#define NUM_PLUGINS 1

extern "C" EXPORT void getLibInfo (Plugin::LibraryInfo* info)
{
    info->apiVersion = PLUGIN_API_VER;
    info->name = "Dual DLCLive Bridge";
    info->libVersion = ProjectInfo::versionString;
    info->numPlugins = NUM_PLUGINS;
}

extern "C" EXPORT int getPluginInfo (int index, Plugin::PluginInfo* info)
{
    switch (index)
    {
        case 0:
            info->type = Plugin::PROCESSOR;
            info->processor.name = "Dual DLCLive Bridge";
            info->processor.type = Plugin::Processor::UTILITY;
            info->processor.creator = &(Plugin::createProcessor<DualDLCLiveBridge>);
            break;
        default:
            return -1;
            break;
    }
    return 0;
}

#ifdef _WIN32
BOOL WINAPI DllMain (IN HINSTANCE hDllHandle,
                     IN DWORD nReason,
                     IN LPVOID Reserved)
{
    return TRUE;
}
#endif
