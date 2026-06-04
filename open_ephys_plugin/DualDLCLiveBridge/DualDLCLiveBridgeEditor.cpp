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
    desiredWidth = 555;

    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enabled", 10, 35);
    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "udp_port", 10, 65);
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "angle_trigger_enabled", 10, 95);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "angle_threshold_deg", 10, 125);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "refractory_ms", 10, 155);

    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "use_filter", 190, 35);
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enable_pcutoff", 190, 65);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "conf_thresh_use", 190, 95);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "conf_thresh_draw", 190, 125);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "median_window", 190, 155);

    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enable_despike", 370, 35);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "despike_threshold_px", 370, 65);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "despike_reset_gap_frames", 370, 95);
    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enable_hold", 370, 125);
    addBoundedValueParameterEditor (Parameter::PROCESSOR_SCOPE, "max_hold_frames", 370, 155);

    statusLabel.setText ("pkts 0 | mode - | pair - | ttl 0x00 | L - | R - | q 0 | age -", dontSendNotification);
    statusLabel.setBounds (10, 185, 535, 20);
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
