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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
    double getLastLeftAngleDeg() const;
    double getLastRightAngleDeg() const;
    int getLastPacketMode() const;

private:
    struct PosePoint
    {
        bool valid = false;
        double x = 0.0;
        double y = 0.0;
        double likelihood = 0.0;
    };

    struct PointFilterState
    {
        bool hasLastGood = false;
        double lastGoodX = 0.0;
        double lastGoodY = 0.0;
        int64 lastGoodFrameId = 0;
        std::deque<double> xHist;
        std::deque<double> yHist;
    };

    struct TripletConfig
    {
        std::array<String, 3> left {{ "hl_hip_l", "hl_ankle_l", "hl_toes_l" }};
        std::array<String, 3> right {{ "hl_hip_r", "hl_ankle_r", "hl_toes_r" }};
    };

    struct SidePoseResult
    {
        bool hasTriplet = false;
        bool hasAngle = false;
        double angleDeg = 0.0;
        String pickedSide;
    };

    using PosePointMap = std::unordered_map<std::string, PosePoint>;
    using FilterStateMap = std::unordered_map<std::string, PointFilterState>;

    void ensureSocket();
    void closeSocket();
    void run() override;
    bool applyMessage (const String& message, String& ackMessage);
    bool applyTtlMessage (const var& parsed, uint8& ttlWord);
    bool applyPoseMessage (const var& parsed, uint8& ttlWord);
    SidePoseResult evaluateSidePose (const var& sideObject,
                                     const String& cameraName,
                                     const TripletConfig& triplets,
                                     FilterStateMap& filterStates);
    TripletConfig readTripletConfig (const var& parsed) const;
    PosePoint readPosePoint (const var& rawPoints, const String& name) const;
    PosePoint filterPoint (const String& name,
                           const PosePoint& rawPoint,
                           int64 frameId,
                           FilterStateMap& filterStates) const;
    std::pair<int, double> scoreTriplet (const PosePointMap& points,
                                         const std::array<String, 3>& triplet) const;
    PosePoint pointFromMap (const PosePointMap& points, const String& name) const;
    bool safeAngleDeg (const PosePoint& a, const PosePoint& b, const PosePoint& c, double& angleDeg) const;
    void resetPoseFilters();
    bool shouldEmitAngleTrigger (int line, bool requested);
    void queueTtlWord (uint8 ttlWord);
    void emitPendingTtlState (int numSamples);
    bool getEnabled() const;
    int getUdpPort() const;
    bool getBoolParam (const String& name, bool fallback) const;
    int getIntParam (const String& name, int fallback) const;
    float getFloatParam (const String& name, float fallback) const;
    static bool readFiniteDouble (const var& value, double& out);
    static std::string pointKey (const String& name);
    static double medianValue (const std::deque<double>& values);

    std::unique_ptr<DatagramSocket> socket;
    CriticalSection socketLock;
    std::atomic<int> currentPort { -1 };
    bool ttlChannelReady = false;
    std::array<bool, 8> emittedLineStates {};
    std::array<std::atomic<bool>, 8> desiredLineStates {};
    CriticalSection pendingLock;
    std::deque<uint8> pendingTtlWords;
    CriticalSection poseStateLock;
    FilterStateMap leftFilterStates;
    FilterStateMap rightFilterStates;
    std::atomic<int64> packetsReceived { 0 };
    std::atomic<int64> lastPairIndex { -1 };
    std::atomic<int64> lastPacketTimeMs { 0 };
    std::atomic<uint8> lastTtlWord { 0 };
    std::atomic<double> lastLeftAngleDeg { -1.0 };
    std::atomic<double> lastRightAngleDeg { -1.0 };
    std::atomic<int> lastPacketMode { 0 };
    std::array<std::atomic<int64>, 8> lastTriggerTimeMs {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DualDLCLiveBridge);
};

#endif
