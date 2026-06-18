/*
    ------------------------------------------------------------------

    Dual DLCLive bridge editor.

    ------------------------------------------------------------------
*/

#include "DualDLCLiveBridgeEditor.h"
#include "DualDLCLiveBridge.h"

DualDLCLiveBridgeEditor::DualDLCLiveBridgeEditor (GenericProcessor* parentNode)
    : GenericEditor (parentNode),
      bridge (static_cast<DualDLCLiveBridge*> (parentNode))
{
    // Compact 5-column x 3-row parameter grid so all 15 params AND the live
    // status line fit inside the ~145px Open Ephys editor strip. The previous
    // 3x5 layout pushed the bottom param row (y=155) and the status label
    // (y=185) below the visible area, so they were clipped off-screen.
    desiredWidth = 790;

    // Row 1 (y=32)
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enabled", 10, 32);
    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "udp_port", 165, 32);
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "angle_trigger_enabled", 320, 32);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "angle_threshold_deg", 475, 32);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "refractory_ms", 630, 32);

    // Row 2 (y=62)
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "use_filter", 10, 62);
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enable_pcutoff", 165, 62);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "conf_thresh_use", 320, 62);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "conf_thresh_draw", 475, 62);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "median_window", 630, 62);

    // Row 3 (y=92)
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enable_despike", 10, 92);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "despike_threshold_px", 165, 92);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "despike_reset_gap_frames", 320, 92);
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enable_hold", 475, 92);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "max_hold_frames", 630, 92);

    // Live status row (y=120) — now inside the visible editor area.
    statusLabel.setText ("pkts 0 | mode - | pair - | ttl 0x00 | L - | R - | q 0 | age -", dontSendNotification);
    statusLabel.setBounds (10, 120, 770, 18);
    statusLabel.setJustificationType (Justification::left);
    addAndMakeVisible (statusLabel);

    startTimer (250);
}

void DualDLCLiveBridgeEditor::timerCallback()
{
    refreshStatus();
}

void DualDLCLiveBridgeEditor::refreshStatus()
{
    if (bridge == nullptr)
        return;

    const int64 packets = bridge->getPacketsReceived();
    const int64 pairIndex = bridge->getLastPairIndex();
    const int64 ageMs = bridge->getLastPacketAgeMs();
    const int pending = bridge->getPendingTtlWordCount();
    const uint8 ttlWord = bridge->getLastTtlWord();
    const int packetMode = bridge->getLastPacketMode();
    const double leftAngle = bridge->getLastLeftAngleDeg();
    const double rightAngle = bridge->getLastRightAngleDeg();

    String pairText = pairIndex >= 0 ? String (pairIndex) : "-";
    String ageText = ageMs >= 0 ? String (ageMs) + "ms" : "-";
    String ttlText = String::toHexString ((int) ttlWord).toUpperCase().paddedLeft ('0', 2);
    String modeText = packetMode == 3 ? "bin" : (packetMode == 2 ? "pose" : (packetMode == 1 ? "ttl" : "-"));
    String leftText = leftAngle >= 0.0 ? String (leftAngle, 1) : "-";
    String rightText = rightAngle >= 0.0 ? String (rightAngle, 1) : "-";

    statusLabel.setText ("pkts " + String (packets)
                             + " | mode " + modeText
                             + " | pair " + pairText
                             + " | ttl 0x" + ttlText
                             + " | L " + leftText
                             + " | R " + rightText
                             + " | q " + String (pending)
                             + " | age " + ageText,
                         dontSendNotification);
}
