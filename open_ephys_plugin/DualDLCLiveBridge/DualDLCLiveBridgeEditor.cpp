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
    desiredWidth = 285;

    addToggleParameterEditor (Parameter::PROCESSOR_SCOPE, "enabled", 10, 35);
    addTextBoxParameterEditor (Parameter::PROCESSOR_SCOPE, "udp_port", 10, 65);

    statusLabel.setText ("pkts 0 | pair - | ttl 0x00 | q 0 | age -", dontSendNotification);
    statusLabel.setBounds (10, 95, 265, 20);
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

    String pairText = pairIndex >= 0 ? String (pairIndex) : "-";
    String ageText = ageMs >= 0 ? String (ageMs) + "ms" : "-";
    String ttlText = String::toHexString ((int) ttlWord).toUpperCase().paddedLeft ('0', 2);

    statusLabel.setText ("pkts " + String (packets)
                             + " | pair " + pairText
                             + " | ttl 0x" + ttlText
                             + " | q " + String (pending)
                             + " | age " + ageText,
                         dontSendNotification);
}
