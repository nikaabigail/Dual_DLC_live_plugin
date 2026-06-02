/*
    ------------------------------------------------------------------

    Dual DLCLive bridge editor.

    ------------------------------------------------------------------
*/

#ifndef __DUALDLCLIVEBRIDGEEDITOR_H__
#define __DUALDLCLIVEBRIDGEEDITOR_H__

#include <EditorHeaders.h>

class DualDLCLiveBridge;

class DualDLCLiveBridgeEditor : public GenericEditor,
                                private Timer
{
public:
    DualDLCLiveBridgeEditor (GenericProcessor* parentNode);
    ~DualDLCLiveBridgeEditor() {}

private:
    void timerCallback() override;
    void refreshStatus();

    DualDLCLiveBridge* bridge = nullptr;
    Label statusLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DualDLCLiveBridgeEditor);
};

#endif
