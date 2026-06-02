/*
    ------------------------------------------------------------------

    Dual DLCLive bridge for Open Ephys GUI.

    ------------------------------------------------------------------
*/

#ifndef __DUALDLCLIVEBRIDGE_H__
#define __DUALDLCLIVEBRIDGE_H__

#include "DualDLCLiveBridgeEditor.h"

#include <ProcessorHeaders.h>

#include <array>
#include <atomic>
#include <deque>
#include <memory>

class DualDLCLiveBridge : public GenericProcessor,
                          private Thread
{
public:
    DualDLCLiveBridge();
    ~DualDLCLiveBridge() override;

    void registerParameters() override;
    void parameterValueChanged (Parameter* param) override;
    void updateSettings() override;
    void process (AudioBuffer<float>& buffer) override;
    AudioProcessorEditor* createEditor() override;

    int getCurrentPort() const;
    int64 getPacketsReceived() const;
    int64 getLastPairIndex() const;
    int64 getLastPacketAgeMs() const;
    int getPendingTtlWordCount() const;
    uint8 getLastTtlWord() const;

private:
    void ensureSocket();
    void closeSocket();
    void run() override;
    bool applyMessage (const String& message, String& ackMessage);
    void queueTtlWord (uint8 ttlWord);
    void emitPendingTtlState (int numSamples);
    bool getEnabled() const;
    int getUdpPort() const;

    std::unique_ptr<DatagramSocket> socket;
    CriticalSection socketLock;
    std::atomic<int> currentPort { -1 };
    bool ttlChannelReady = false;
    std::array<bool, 8> emittedLineStates {};
    std::array<std::atomic<bool>, 8> desiredLineStates {};
    CriticalSection pendingLock;
    std::deque<uint8> pendingTtlWords;
    std::atomic<int64> packetsReceived { 0 };
    std::atomic<int64> lastPairIndex { -1 };
    std::atomic<int64> lastPacketTimeMs { 0 };
    std::atomic<uint8> lastTtlWord { 0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DualDLCLiveBridge);
};

#endif
