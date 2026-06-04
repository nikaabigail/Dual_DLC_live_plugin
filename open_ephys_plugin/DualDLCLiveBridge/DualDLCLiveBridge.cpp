/*
    ------------------------------------------------------------------

    Dual DLCLive bridge for Open Ephys GUI.

    Receives dual DLCLive UDP JSON packets. Legacy packets can still provide
    ttl_lines, while pose.v1 packets provide raw pose points and let this
    processor compute validity, angle triggers and TTL states.

    ------------------------------------------------------------------
*/

#include "DualDLCLiveBridge.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace
{
constexpr size_t maxPendingTtlWords = 512;
constexpr int packetModeNone = 0;
constexpr int packetModeLegacyTtl = 1;
constexpr int packetModePose = 2;
constexpr int packetModeBinaryPose = 3;
constexpr std::uint16_t binaryPoseVersion = 1;
constexpr std::uint16_t binaryFlagAck = 1 << 0;

template <typename Value>
bool readBinaryValue (const char* data, int numBytes, size_t& offset, Value& out)
{
    if (offset + sizeof (Value) > (size_t) numBytes)
        return false;

    std::memcpy (&out, data + offset, sizeof (Value));
    offset += sizeof (Value);
    return true;
}
}

DualDLCLiveBridge::DualDLCLiveBridge()
    : GenericProcessor ("Dual DLCLive Bridge"),
      Thread ("Dual DLCLive Bridge UDP")
{
    emittedLineStates.fill (false);
    for (auto& line : desiredLineStates)
        line.store (false);
    for (auto& line : lastTriggerTimeMs)
        line.store (0);
}

DualDLCLiveBridge::~DualDLCLiveBridge()
{
    closeSocket();
}

void DualDLCLiveBridge::registerParameters()
{
    addBooleanParameter (Parameter::PROCESSOR_SCOPE,
                         "enabled",
                         "Enabled",
                         "Listen for dual DLCLive UDP packets",
                         true,
                         false);

    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "udp_port",
                     "UDP port",
                     "Local UDP port used by dual_rt_dlc_live.py",
                     47000,
                     1024,
                     65535,
                     true);

    addBooleanParameter (Parameter::PROCESSOR_SCOPE,
                         "angle_trigger_enabled",
                         "Angle trigger",
                         "Enable angle threshold output on TTL lines 2 and 3",
                         false,
                         false);

    addFloatParameter (Parameter::PROCESSOR_SCOPE,
                       "angle_threshold_deg",
                       "Angle threshold",
                       "Hind angle threshold for TTL trigger lines",
                       "deg",
                       55.0f,
                       0.0f,
                       180.0f,
                       0.1f,
                       false);

    addFloatParameter (Parameter::PROCESSOR_SCOPE,
                       "conf_thresh_use",
                       "Use conf",
                       "Likelihood threshold used by the online pose filter",
                       "",
                       0.20f,
                       0.0f,
                       1.0f,
                       0.01f,
                       false);

    addFloatParameter (Parameter::PROCESSOR_SCOPE,
                       "conf_thresh_draw",
                       "Draw conf",
                       "Likelihood threshold used to accept a visible triplet",
                       "",
                       0.15f,
                       0.0f,
                       1.0f,
                       0.01f,
                       false);

    addBooleanParameter (Parameter::PROCESSOR_SCOPE,
                         "use_filter",
                         "Filter",
                         "Apply the same online filtering used by dual_rt_dlc_live.py",
                         true,
                         false);

    addBooleanParameter (Parameter::PROCESSOR_SCOPE,
                         "enable_pcutoff",
                         "P cutoff",
                         "Drop points below the use confidence threshold before filtering",
                         true,
                         false);

    addBooleanParameter (Parameter::PROCESSOR_SCOPE,
                         "enable_despike",
                         "Despike",
                         "Reject sudden implausible point jumps",
                         true,
                         false);

    addFloatParameter (Parameter::PROCESSOR_SCOPE,
                       "despike_threshold_px",
                       "Despike px",
                       "Maximum accepted point jump unless reacquiring after a gap",
                       "px",
                       150.0f,
                       1.0f,
                       2000.0f,
                       1.0f,
                       false);

    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "despike_reset_gap_frames",
                     "Reset gap",
                     "Frame gap after which the despike filter allows reacquisition",
                     15,
                     0,
                     10000,
                     false);

    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "median_window",
                     "Median",
                     "Median filter window size in frames",
                     3,
                     1,
                     31,
                     false);

    addBooleanParameter (Parameter::PROCESSOR_SCOPE,
                         "enable_hold",
                         "Hold",
                         "Hold the last good point for a limited number of frames",
                         false,
                         false);

    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "max_hold_frames",
                     "Hold frames",
                     "Maximum number of frames to hold the last good point",
                     20,
                     0,
                     10000,
                     false);

    addIntParameter (Parameter::PROCESSOR_SCOPE,
                     "refractory_ms",
                     "Refractory",
                     "Minimum time between angle trigger rising edges",
                     0,
                     0,
                     60000,
                     false);
}

AudioProcessorEditor* DualDLCLiveBridge::createEditor()
{
    editor = std::make_unique<DualDLCLiveBridgeEditor> (this);
    return editor.get();
}

void DualDLCLiveBridge::updateSettings()
{
    if (eventChannels.size() == 0)
        addTTLChannel ("Dual DLCLive TTL");

    ttlChannelReady = eventChannels.size() > 0;
    ensureSocket();
}

void DualDLCLiveBridge::parameterValueChanged (Parameter* param)
{
    const String name = param->getName();
    if (name.equalsIgnoreCase ("enabled")
        || name.equalsIgnoreCase ("udp_port"))
    {
        ensureSocket();
    }

    if (name.equalsIgnoreCase ("use_filter")
        || name.equalsIgnoreCase ("enable_pcutoff")
        || name.equalsIgnoreCase ("enable_despike")
        || name.equalsIgnoreCase ("despike_threshold_px")
        || name.equalsIgnoreCase ("despike_reset_gap_frames")
        || name.equalsIgnoreCase ("median_window")
        || name.equalsIgnoreCase ("enable_hold")
        || name.equalsIgnoreCase ("max_hold_frames"))
    {
        resetPoseFilters();
    }
}

void DualDLCLiveBridge::process (AudioBuffer<float>& buffer)
{
    checkForEvents();

    if (! ttlChannelReady)
        return;

    emitPendingTtlState (buffer.getNumSamples());
}

int DualDLCLiveBridge::getCurrentPort() const
{
    return currentPort.load();
}

int64 DualDLCLiveBridge::getPacketsReceived() const
{
    return packetsReceived.load();
}

int64 DualDLCLiveBridge::getLastPairIndex() const
{
    return lastPairIndex.load();
}

int64 DualDLCLiveBridge::getLastPacketAgeMs() const
{
    const int64 lastTime = lastPacketTimeMs.load();
    if (lastTime <= 0)
        return -1;

    return Time::currentTimeMillis() - lastTime;
}

uint8 DualDLCLiveBridge::getLastTtlWord() const
{
    return lastTtlWord.load();
}

double DualDLCLiveBridge::getLastLeftAngleDeg() const
{
    return lastLeftAngleDeg.load();
}

double DualDLCLiveBridge::getLastRightAngleDeg() const
{
    return lastRightAngleDeg.load();
}

int DualDLCLiveBridge::getLastPacketMode() const
{
    return lastPacketMode.load();
}

int DualDLCLiveBridge::getPendingTtlWordCount() const
{
    const ScopedLock lock (pendingLock);
    return (int) pendingTtlWords.size();
}

bool DualDLCLiveBridge::getEnabled() const
{
    return bool (getParameter ("enabled")->getValue());
}

int DualDLCLiveBridge::getUdpPort() const
{
    return int (getParameter ("udp_port")->getValue());
}

void DualDLCLiveBridge::ensureSocket()
{
    if (! getEnabled())
    {
        closeSocket();
        return;
    }

    const int requestedPort = getUdpPort();
    if (isThreadRunning() && currentPort.load() == requestedPort)
        return;

    closeSocket();

    const ScopedLock lock (socketLock);
    socket.reset (new DatagramSocket (false));
    socket->setEnablePortReuse (true);

    if (socket->bindToPort (requestedPort, "127.0.0.1"))
    {
        currentPort.store (requestedPort);
        startThread();
        CoreServices::sendStatusMessage ("Dual DLCLive Bridge listening on UDP 127.0.0.1:"
                                         + String (requestedPort));
    }
    else
    {
        socket.reset();
        currentPort.store (-1);
        CoreServices::sendStatusMessage ("Dual DLCLive Bridge could not bind UDP port "
                                         + String (requestedPort));
    }
}

void DualDLCLiveBridge::closeSocket()
{
    signalThreadShouldExit();

    {
        const ScopedLock lock (socketLock);
        if (socket != nullptr)
            socket->shutdown();
    }

    stopThread (500);

    {
        const ScopedLock lock (socketLock);
        socket.reset();
    }
    queueTtlWord (0);
    for (auto& line : desiredLineStates)
        line.store (false);
    resetPoseFilters();
    lastPacketMode.store (packetModeNone);
    lastLeftAngleDeg.store (-1.0);
    lastRightAngleDeg.store (-1.0);
    currentPort.store (-1);
}

void DualDLCLiveBridge::run()
{
    constexpr int maxPacketsPerWake = 64;
    constexpr int maxPacketBytes = 32768;
    char buffer[maxPacketBytes] {};

    while (! threadShouldExit())
    {
        for (int packet = 0; packet < maxPacketsPerWake && ! threadShouldExit(); packet++)
        {
            String senderAddress;
            int senderPort = 0;
            int bytesRead = 0;

            {
                const ScopedLock lock (socketLock);
                if (socket == nullptr)
                    break;

                const int timeoutMs = packet == 0 ? 100 : 0;
                if (socket->waitUntilReady (true, timeoutMs) <= 0)
                    break;

                bytesRead = socket->read (buffer, maxPacketBytes, false, senderAddress, senderPort);
                if (bytesRead <= 0)
                    break;
            }

            if (bytesRead > 0)
            {
                String ackMessage;
                if (applyDatagram (buffer, bytesRead, ackMessage) && ackMessage.isNotEmpty() && senderPort > 0)
                {
                    const ScopedLock lock (socketLock);
                    if (socket != nullptr)
                    {
                        const auto utf8 = ackMessage.toRawUTF8();
                        socket->write (senderAddress, senderPort, utf8, (int) ackMessage.getNumBytesAsUTF8());
                    }
                }
            }
        }
    }
}

bool DualDLCLiveBridge::applyMessage (const String& message, String& ackMessage)
{
    var parsed;
    if (JSON::parse (message, parsed).failed() || ! parsed.isObject())
        return false;

    const String schema = parsed.getProperty ("schema", var()).toString();
    uint8 ttlWord = 0;

    if (schema.isEmpty() || schema == "dual_dlc_live.v1")
    {
        if (! applyTtlMessage (parsed, ttlWord))
            return false;
        lastPacketMode.store (packetModeLegacyTtl);
    }
    else if (schema == "dual_dlc_live.pose.v1")
    {
        if (! applyPoseMessage (parsed, ttlWord))
            return false;
        lastPacketMode.store (packetModePose);
    }
    else
    {
        return false;
    }

    queueTtlWord (ttlWord);
    packetsReceived.fetch_add (1);
    const int64 pairIndex = (int64) parsed.getProperty ("pair_index", var (-1));
    lastPairIndex.store (pairIndex);
    lastPacketTimeMs.store (Time::currentTimeMillis());

    const bool requestAck = bool (parsed.getProperty ("ack", var (false)))
                            || bool (parsed.getProperty ("request_ack", var (false)));
    if (requestAck)
    {
        String ttlHex = String::toHexString ((int) ttlWord).toUpperCase();
        if (ttlHex.length() < 2)
            ttlHex = "0" + ttlHex;

        const String modeText = lastPacketMode.load() == packetModePose ? "pose" : "ttl";
        String angleText;
        if (lastPacketMode.load() == packetModePose)
        {
            const double leftAngle = lastLeftAngleDeg.load();
            const double rightAngle = lastRightAngleDeg.load();
            angleText = " left_angle=" + (leftAngle >= 0.0 ? String (leftAngle, 2) : String ("nan"))
                        + " right_angle=" + (rightAngle >= 0.0 ? String (rightAngle, 2) : String ("nan"));
        }

        ackMessage = "dual_dlc_live.ack pair=" + String (pairIndex)
                     + " mode=" + modeText
                     + " ttl=0x" + ttlHex
                     + angleText
                     + "\n";
    }
    return true;
}

bool DualDLCLiveBridge::applyDatagram (const char* data, int numBytes, String& ackMessage)
{
    if (data == nullptr || numBytes <= 0)
        return false;

    if (numBytes >= 4 && std::memcmp (data, "DDLP", 4) == 0)
    {
        uint8 ttlWord = 0;
        int64 pairIndex = -1;
        bool requestAck = false;
        if (! applyBinaryPosePacket (data, numBytes, ttlWord, pairIndex, requestAck))
            return false;

        lastPacketMode.store (packetModeBinaryPose);
        queueTtlWord (ttlWord);
        packetsReceived.fetch_add (1);
        lastPairIndex.store (pairIndex);
        lastPacketTimeMs.store (Time::currentTimeMillis());

        if (requestAck)
        {
            String ttlHex = String::toHexString ((int) ttlWord).toUpperCase();
            if (ttlHex.length() < 2)
                ttlHex = "0" + ttlHex;

            const double leftAngle = lastLeftAngleDeg.load();
            const double rightAngle = lastRightAngleDeg.load();
            ackMessage = "dual_dlc_live.ack pair=" + String (pairIndex)
                         + " mode=binary"
                         + " ttl=0x" + ttlHex
                         + " left_angle=" + (leftAngle >= 0.0 ? String (leftAngle, 2) : String ("nan"))
                         + " right_angle=" + (rightAngle >= 0.0 ? String (rightAngle, 2) : String ("nan"))
                         + "\n";
        }
        return true;
    }

    const String message = String::fromUTF8 (data, numBytes).trim();
    return message.isNotEmpty() && applyMessage (message, ackMessage);
}

bool DualDLCLiveBridge::applyTtlMessage (const var& parsed, uint8& ttlWord)
{
    const var ttlLines = parsed.getProperty ("ttl_lines", var());
    if (! ttlLines.isArray())
        return false;

    Array<var>* lines = ttlLines.getArray();
    const int nLines = jmin (8, lines->size());
    std::array<bool, 8> nextStates {};
    for (int line = 0; line < 8; line++)
        nextStates[(size_t) line] = desiredLineStates[(size_t) line].load();

    for (int line = 0; line < nLines; line++)
        nextStates[(size_t) line] = bool (lines->getReference (line));

    ttlWord = 0;
    for (int line = 0; line < 8; line++)
    {
        desiredLineStates[(size_t) line].store (nextStates[(size_t) line]);
        if (nextStates[(size_t) line])
            ttlWord |= static_cast<uint8> (1 << line);
    }

    double angle = 0.0;
    const var left = parsed.getProperty ("left", var());
    const var right = parsed.getProperty ("right", var());
    lastLeftAngleDeg.store (left.isObject() && readFiniteDouble (left.getProperty ("angle_deg", var()), angle) ? angle : -1.0);
    lastRightAngleDeg.store (right.isObject() && readFiniteDouble (right.getProperty ("angle_deg", var()), angle) ? angle : -1.0);
    return true;
}

bool DualDLCLiveBridge::applyPoseMessage (const var& parsed, uint8& ttlWord)
{
    const ScopedLock lock (poseStateLock);

    const TripletConfig triplets = readTripletConfig (parsed);
    SidePoseResult left = evaluateSidePose (parsed.getProperty ("left", var()),
                                            "left",
                                            triplets,
                                            leftFilterStates);
    SidePoseResult right = evaluateSidePose (parsed.getProperty ("right", var()),
                                             "right",
                                             triplets,
                                             rightFilterStates);

    const bool angleEnabled = getBoolParam ("angle_trigger_enabled", false);
    const double angleThreshold = (double) getFloatParam ("angle_threshold_deg", 55.0f);

    std::array<bool, 8> nextStates {};
    nextStates[0] = left.hasTriplet;
    nextStates[1] = right.hasTriplet;
    nextStates[2] = shouldEmitAngleTrigger (2, angleEnabled && left.hasAngle && left.angleDeg <= angleThreshold);
    nextStates[3] = shouldEmitAngleTrigger (3, angleEnabled && right.hasAngle && right.angleDeg <= angleThreshold);

    ttlWord = 0;
    for (int line = 0; line < 8; line++)
    {
        desiredLineStates[(size_t) line].store (nextStates[(size_t) line]);
        if (nextStates[(size_t) line])
            ttlWord |= static_cast<uint8> (1 << line);
    }

    lastLeftAngleDeg.store (left.hasAngle ? left.angleDeg : -1.0);
    lastRightAngleDeg.store (right.hasAngle ? right.angleDeg : -1.0);
    return true;
}

bool DualDLCLiveBridge::applyBinaryPosePacket (const char* data,
                                               int numBytes,
                                               uint8& ttlWord,
                                               int64& pairIndex,
                                               bool& requestAck)
{
    size_t offset = 4;
    std::uint16_t version = 0;
    std::uint16_t flags = 0;
    std::int64_t packetPairIndex = -1;
    double hostTime = 0.0;
    float hostDtMs = 0.0f;
    float cameraDtMs = 0.0f;
    std::uint16_t pointCount = 0;
    std::uint16_t reserved = 0;

    if (! readBinaryValue (data, numBytes, offset, version)
        || ! readBinaryValue (data, numBytes, offset, flags)
        || ! readBinaryValue (data, numBytes, offset, packetPairIndex)
        || ! readBinaryValue (data, numBytes, offset, hostTime)
        || ! readBinaryValue (data, numBytes, offset, hostDtMs)
        || ! readBinaryValue (data, numBytes, offset, cameraDtMs)
        || ! readBinaryValue (data, numBytes, offset, pointCount)
        || ! readBinaryValue (data, numBytes, offset, reserved))
    {
        return false;
    }

    ignoreUnused (hostTime, hostDtMs, cameraDtMs, reserved);
    if (version != binaryPoseVersion)
        return false;

    const std::array<String, 6> pointNames {{
        "hl_ankle_l",
        "hl_ankle_r",
        "hl_hip_l",
        "hl_hip_r",
        "hl_toes_l",
        "hl_toes_r",
    }};
    if (pointCount != pointNames.size())
        return false;

    struct BinarySide
    {
        int64 frameId = 0;
        PosePointMap rawPoints;
    };

    auto readSide = [&] (BinarySide& side) -> bool
    {
        std::int64_t frameId = 0;
        std::int64_t sourceFrameId = 0;
        double captureTs = 0.0;
        float inferMs = 0.0f;
        std::uint32_t drops = 0;
        std::uint16_t rawVisible = 0;
        std::uint16_t sideReserved = 0;

        if (! readBinaryValue (data, numBytes, offset, frameId)
            || ! readBinaryValue (data, numBytes, offset, sourceFrameId)
            || ! readBinaryValue (data, numBytes, offset, captureTs)
            || ! readBinaryValue (data, numBytes, offset, inferMs)
            || ! readBinaryValue (data, numBytes, offset, drops)
            || ! readBinaryValue (data, numBytes, offset, rawVisible)
            || ! readBinaryValue (data, numBytes, offset, sideReserved))
        {
            return false;
        }

        ignoreUnused (sourceFrameId, captureTs, inferMs, drops, rawVisible, sideReserved);
        side.frameId = (int64) frameId;
        for (size_t index = 0; index < pointNames.size(); index++)
        {
            float x = 0.0f;
            float y = 0.0f;
            float likelihood = 0.0f;
            if (! readBinaryValue (data, numBytes, offset, x)
                || ! readBinaryValue (data, numBytes, offset, y)
                || ! readBinaryValue (data, numBytes, offset, likelihood))
            {
                return false;
            }

            PosePoint point;
            if (std::isfinite (x) && std::isfinite (y) && std::isfinite (likelihood))
            {
                point.valid = true;
                point.x = (double) x;
                point.y = (double) y;
                point.likelihood = (double) likelihood;
            }
            side.rawPoints[pointKey (pointNames[index])] = point;
        }
        return true;
    };

    BinarySide leftSide;
    BinarySide rightSide;
    if (! readSide (leftSide) || ! readSide (rightSide))
        return false;

    const ScopedLock lock (poseStateLock);
    const TripletConfig triplets;
    SidePoseResult left = evaluateSidePosePoints (leftSide.rawPoints,
                                                  leftSide.frameId,
                                                  "left",
                                                  triplets,
                                                  leftFilterStates);
    SidePoseResult right = evaluateSidePosePoints (rightSide.rawPoints,
                                                   rightSide.frameId,
                                                   "right",
                                                   triplets,
                                                   rightFilterStates);

    const bool angleEnabled = getBoolParam ("angle_trigger_enabled", false);
    const double angleThreshold = (double) getFloatParam ("angle_threshold_deg", 55.0f);

    std::array<bool, 8> nextStates {};
    nextStates[0] = left.hasTriplet;
    nextStates[1] = right.hasTriplet;
    nextStates[2] = shouldEmitAngleTrigger (2, angleEnabled && left.hasAngle && left.angleDeg <= angleThreshold);
    nextStates[3] = shouldEmitAngleTrigger (3, angleEnabled && right.hasAngle && right.angleDeg <= angleThreshold);

    ttlWord = 0;
    for (int line = 0; line < 8; line++)
    {
        desiredLineStates[(size_t) line].store (nextStates[(size_t) line]);
        if (nextStates[(size_t) line])
            ttlWord |= static_cast<uint8> (1 << line);
    }

    lastLeftAngleDeg.store (left.hasAngle ? left.angleDeg : -1.0);
    lastRightAngleDeg.store (right.hasAngle ? right.angleDeg : -1.0);
    pairIndex = (int64) packetPairIndex;
    requestAck = (flags & binaryFlagAck) != 0;
    return true;
}

DualDLCLiveBridge::SidePoseResult DualDLCLiveBridge::evaluateSidePose (const var& sideObject,
                                                                        const String& cameraName,
                                                                        const TripletConfig& triplets,
                                                                        FilterStateMap& filterStates)
{
    SidePoseResult result;
    result.pickedSide = cameraName;

    if (! sideObject.isObject())
        return result;

    const var rawPoints = sideObject.getProperty ("raw_points", var());
    if (! rawPoints.isObject())
        return result;

    const int64 frameId = (int64) sideObject.getProperty ("frame_id", var (0));
    std::vector<String> pointNames;

    auto addTripletNames = [&pointNames] (const std::array<String, 3>& triplet)
    {
        for (const String& name : triplet)
        {
            bool exists = false;
            for (const String& existing : pointNames)
            {
                if (existing == name)
                {
                    exists = true;
                    break;
                }
            }
            if (! exists)
                pointNames.push_back (name);
        }
    };

    addTripletNames (triplets.left);
    addTripletNames (triplets.right);

    PosePointMap rawPointMap;
    for (const String& name : pointNames)
        rawPointMap[pointKey (name)] = readPosePoint (rawPoints, name);

    return evaluateSidePosePoints (rawPointMap, frameId, cameraName, triplets, filterStates);
}

DualDLCLiveBridge::SidePoseResult DualDLCLiveBridge::evaluateSidePosePoints (const PosePointMap& rawPoints,
                                                                              int64 frameId,
                                                                              const String& cameraName,
                                                                              const TripletConfig& triplets,
                                                                              FilterStateMap& filterStates)
{
    SidePoseResult result;
    result.pickedSide = cameraName;

    std::vector<String> pointNames;
    auto addTripletNames = [&pointNames] (const std::array<String, 3>& triplet)
    {
        for (const String& name : triplet)
        {
            bool exists = false;
            for (const String& existing : pointNames)
            {
                if (existing == name)
                {
                    exists = true;
                    break;
                }
            }
            if (! exists)
                pointNames.push_back (name);
        }
    };

    addTripletNames (triplets.left);
    addTripletNames (triplets.right);

    PosePointMap filteredPoints;
    for (const String& name : pointNames)
        filteredPoints[pointKey (name)] = filterPoint (name, pointFromMap (rawPoints, name), frameId, filterStates);

    const auto leftScore = scoreTriplet (filteredPoints, triplets.left);
    const auto rightScore = scoreTriplet (filteredPoints, triplets.right);
    const bool useRight = rightScore.first > leftScore.first
                          || (rightScore.first == leftScore.first && rightScore.second > leftScore.second);
    const std::array<String, 3>& selected = useRight ? triplets.right : triplets.left;
    result.pickedSide = useRight ? "right" : "left";

    const PosePoint hip = pointFromMap (filteredPoints, selected[0]);
    const PosePoint ankle = pointFromMap (filteredPoints, selected[1]);
    const PosePoint toes = pointFromMap (filteredPoints, selected[2]);
    const double confDraw = (double) getFloatParam ("conf_thresh_draw", 0.15f);

    result.hasTriplet = hip.valid
                        && ankle.valid
                        && toes.valid
                        && hip.likelihood >= confDraw
                        && ankle.likelihood >= confDraw
                        && toes.likelihood >= confDraw;

    if (result.hasTriplet)
        result.hasAngle = safeAngleDeg (hip, ankle, toes, result.angleDeg);

    return result;
}

DualDLCLiveBridge::TripletConfig DualDLCLiveBridge::readTripletConfig (const var& parsed) const
{
    TripletConfig triplets;
    const var sideSets = parsed.getProperty ("side_point_sets", var());
    if (! sideSets.isObject())
        return triplets;

    auto readTriplet = [&sideSets] (const String& sideName, std::array<String, 3>& destination)
    {
        const var names = sideSets.getProperty (Identifier (sideName), var());
        if (! names.isArray())
            return;

        Array<var>* array = names.getArray();
        if (array->size() != 3)
            return;

        for (int i = 0; i < 3; i++)
            destination[(size_t) i] = array->getReference (i).toString();
    };

    readTriplet ("left", triplets.left);
    readTriplet ("right", triplets.right);
    return triplets;
}

DualDLCLiveBridge::PosePoint DualDLCLiveBridge::readPosePoint (const var& rawPoints, const String& name) const
{
    PosePoint point;
    if (! rawPoints.isObject())
        return point;

    const var rawPoint = rawPoints.getProperty (Identifier (name), var());
    if (! rawPoint.isObject())
        return point;

    double x = 0.0;
    double y = 0.0;
    double likelihood = 0.0;
    if (! readFiniteDouble (rawPoint.getProperty ("x", var()), x)
        || ! readFiniteDouble (rawPoint.getProperty ("y", var()), y)
        || ! readFiniteDouble (rawPoint.getProperty ("likelihood", var()), likelihood))
    {
        return point;
    }

    point.valid = true;
    point.x = x;
    point.y = y;
    point.likelihood = likelihood;
    return point;
}

DualDLCLiveBridge::PosePoint DualDLCLiveBridge::filterPoint (const String& name,
                                                             const PosePoint& rawPoint,
                                                             int64 frameId,
                                                             FilterStateMap& filterStates) const
{
    if (! getBoolParam ("use_filter", true))
        return rawPoint;

    PointFilterState& state = filterStates[pointKey (name)];
    bool isGood = rawPoint.valid
                  && ((! getBoolParam ("enable_pcutoff", true))
                      || rawPoint.likelihood >= (double) getFloatParam ("conf_thresh_use", 0.20f));

    if (isGood && getBoolParam ("enable_despike", true) && state.hasLastGood)
    {
        const double jump = std::hypot (rawPoint.x - state.lastGoodX, rawPoint.y - state.lastGoodY);
        const int64 gap = frameId - state.lastGoodFrameId;
        const bool allowReacquire = gap > (int64) getIntParam ("despike_reset_gap_frames", 15);
        if (jump > (double) getFloatParam ("despike_threshold_px", 150.0f) && ! allowReacquire)
            isGood = false;
    }

    if (isGood)
    {
        const int medianWindow = jmax (1, getIntParam ("median_window", 3));
        state.hasLastGood = true;
        state.lastGoodX = rawPoint.x;
        state.lastGoodY = rawPoint.y;
        state.lastGoodFrameId = frameId;
        state.xHist.push_back (rawPoint.x);
        state.yHist.push_back (rawPoint.y);
        while ((int) state.xHist.size() > medianWindow)
            state.xHist.pop_front();
        while ((int) state.yHist.size() > medianWindow)
            state.yHist.pop_front();

        PosePoint filtered = rawPoint;
        filtered.x = medianValue (state.xHist);
        filtered.y = medianValue (state.yHist);
        return filtered;
    }

    if (getBoolParam ("enable_hold", false) && state.hasLastGood)
    {
        const int64 gap = frameId - state.lastGoodFrameId;
        if (gap <= (int64) getIntParam ("max_hold_frames", 20))
        {
            PosePoint held;
            held.valid = true;
            held.x = state.lastGoodX;
            held.y = state.lastGoodY;
            held.likelihood = jmin (1.0, (double) jmax (getFloatParam ("conf_thresh_draw", 0.15f),
                                                       getFloatParam ("conf_thresh_use", 0.20f))
                                         + 0.01);
            return held;
        }
    }

    return PosePoint {};
}

std::pair<int, double> DualDLCLiveBridge::scoreTriplet (const PosePointMap& points,
                                                        const std::array<String, 3>& triplet) const
{
    const double confDraw = (double) getFloatParam ("conf_thresh_draw", 0.15f);
    int count = 0;
    double likelihoodSum = 0.0;
    for (const String& name : triplet)
    {
        const PosePoint point = pointFromMap (points, name);
        if (point.valid && point.likelihood >= confDraw)
        {
            count++;
            likelihoodSum += point.likelihood;
        }
    }
    return { count, likelihoodSum };
}

DualDLCLiveBridge::PosePoint DualDLCLiveBridge::pointFromMap (const PosePointMap& points, const String& name) const
{
    const auto it = points.find (pointKey (name));
    return it == points.end() ? PosePoint {} : it->second;
}

bool DualDLCLiveBridge::safeAngleDeg (const PosePoint& a,
                                      const PosePoint& b,
                                      const PosePoint& c,
                                      double& angleDeg) const
{
    const double bax = a.x - b.x;
    const double bay = a.y - b.y;
    const double bcx = c.x - b.x;
    const double bcy = c.y - b.y;
    const double n1 = std::hypot (bax, bay);
    const double n2 = std::hypot (bcx, bcy);
    if (n1 < 1.0e-6 || n2 < 1.0e-6)
        return false;

    const double cosValue = jlimit (-1.0, 1.0, (bax * bcx + bay * bcy) / (n1 * n2));
    angleDeg = std::acos (cosValue) * 180.0 / double_Pi;
    return std::isfinite (angleDeg);
}

void DualDLCLiveBridge::resetPoseFilters()
{
    const ScopedLock lock (poseStateLock);
    leftFilterStates.clear();
    rightFilterStates.clear();
    for (auto& line : lastTriggerTimeMs)
        line.store (0);
}

bool DualDLCLiveBridge::shouldEmitAngleTrigger (int line, bool requested)
{
    if (! requested)
        return false;

    const int refractoryMs = getIntParam ("refractory_ms", 0);
    if (refractoryMs <= 0)
    {
        lastTriggerTimeMs[(size_t) line].store (Time::currentTimeMillis());
        return true;
    }

    if (desiredLineStates[(size_t) line].load())
        return true;

    const int64 now = Time::currentTimeMillis();
    const int64 last = lastTriggerTimeMs[(size_t) line].load();
    if (last > 0 && now - last < refractoryMs)
        return false;

    lastTriggerTimeMs[(size_t) line].store (now);
    return true;
}

bool DualDLCLiveBridge::getBoolParam (const String& name, bool fallback) const
{
    if (auto* param = getParameter (name))
        return bool (param->getValue());
    return fallback;
}

int DualDLCLiveBridge::getIntParam (const String& name, int fallback) const
{
    if (auto* param = getParameter (name))
        return int (param->getValue());
    return fallback;
}

float DualDLCLiveBridge::getFloatParam (const String& name, float fallback) const
{
    if (auto* param = getParameter (name))
        return float (param->getValue());
    return fallback;
}

bool DualDLCLiveBridge::readFiniteDouble (const var& value, double& out)
{
    if (value.isVoid() || value.isUndefined())
        return false;
    if (! (value.isDouble() || value.isInt() || value.isInt64()))
        return false;

    out = (double) value;
    return std::isfinite (out);
}

std::string DualDLCLiveBridge::pointKey (const String& name)
{
    return std::string (name.toRawUTF8());
}

double DualDLCLiveBridge::medianValue (const std::deque<double>& values)
{
    if (values.empty())
        return 0.0;

    std::vector<double> sorted (values.begin(), values.end());
    std::sort (sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    if ((n % 2) != 0)
        return sorted[n / 2];

    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

void DualDLCLiveBridge::queueTtlWord (uint8 ttlWord)
{
    const uint8 previousWord = lastTtlWord.exchange (ttlWord);
    if (previousWord == ttlWord)
        return;

    {
        const ScopedLock lock (pendingLock);
        pendingTtlWords.push_back (ttlWord);
        while (pendingTtlWords.size() > maxPendingTtlWords)
            pendingTtlWords.pop_front();
    }
}

void DualDLCLiveBridge::emitPendingTtlState (int numSamples)
{
    std::deque<uint8> words;
    {
        const ScopedLock lock (pendingLock);
        words.swap (pendingTtlWords);
    }

    if (words.empty())
        return;

    const int safeNumSamples = jmax (1, numSamples);
    const int lastSampleIndex = safeNumSamples - 1;
    const int denominator = jmax (1, (int) words.size() - 1);
    int wordIndex = 0;

    for (const uint8 word : words)
    {
        const int sampleIndex = words.size() == 1
                                    ? 0
                                    : jlimit (0, lastSampleIndex, roundToInt ((double) wordIndex * (double) lastSampleIndex / (double) denominator));

        for (int line = 0; line < 8; line++)
        {
            const bool nextState = (word & static_cast<uint8> (1 << line)) != 0;
            if (nextState != emittedLineStates[(size_t) line])
            {
                setTTLState (sampleIndex, line, nextState);
                emittedLineStates[(size_t) line] = nextState;
            }
        }

        wordIndex++;
    }
}
