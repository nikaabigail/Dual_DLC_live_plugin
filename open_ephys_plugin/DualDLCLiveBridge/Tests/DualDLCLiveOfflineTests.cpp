// Native parser/filter regression and file replay. No sockets or process().
#include "gtest/gtest.h"
#include "../DualDLCLiveBridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace
{
using Point = std::array<float, 3>;
using Points = std::array<Point, 8>;
using Snapshot = DualDLCLiveBridge::NativePoseSnapshot;
const std::array<std::string, 8> names {{ "hl_ankle_l", "hl_ankle_r", "hl_hip_l", "hl_hip_r",
                                         "hl_toes_l", "hl_toes_r", "hl_knee_l", "hl_knee_r" }};

std::unique_ptr<DualDLCLiveBridge> makeOfflineProcessor()
{
    auto processor = std::make_unique<DualDLCLiveBridge>();
    processor->setProcessorType (Plugin::Processor::UTILITY);
    processor->setHeadlessMode (true);
    processor->setNodeId (998);
    processor->registerParameters();
    processor->initialize (false);
    // Deliberately no updateSettings(): that is where the UDP listener starts.
    return processor;
}

void setParam (DualDLCLiveBridge& processor, const String& name, var value)
{
    auto* param = processor.getParameter (name);
    ASSERT_NE (param, nullptr) << name.toStdString();
    param->currentValue = value;
    processor.parameterValueChanged (param);
}

template <typename Value>
void append (std::vector<char>& bytes, Value value)
{
    const auto* start = reinterpret_cast<const char*> (&value);
    bytes.insert (bytes.end(), start, start + sizeof (Value));
}

Points samplePoints()
{
    return {{{110, 120, .95f}, {210, 120, .4f}, {100, 60, .95f}, {200, 60, .4f},
             {120, 140, .95f}, {220, 140, .4f}, {105, 90, .95f}, {205, 90, .4f}}};
}

std::vector<char> packet (int count, int64 frame, const Points& left, const Points& right)
{
    std::vector<char> bytes {'D', 'D', 'L', 'P'};
    append<std::uint16_t> (bytes, 1);
    append<std::uint16_t> (bytes, 0);
    append<std::int64_t> (bytes, frame);
    append<double> (bytes, frame / 100.0);
    append<float> (bytes, 0);
    append<float> (bytes, 0);
    append<std::uint16_t> (bytes, (std::uint16_t) count);
    append<std::uint16_t> (bytes, 0);
    for (const auto& side : { left, right })
    {
        append<std::int64_t> (bytes, frame);
        append<std::int64_t> (bytes, frame);
        append<double> (bytes, frame / 100.0);
        append<float> (bytes, 1);
        append<std::uint32_t> (bytes, 0);
        append<std::uint16_t> (bytes, (std::uint16_t) count);
        append<std::uint16_t> (bytes, 0);
        for (int i = 0; i < count; ++i)
            for (float value : side[(size_t) i])
                append<float> (bytes, value);
    }
    return bytes;
}

Snapshot feed (DualDLCLiveBridge& processor, int64 frame, const Points& points, int count = 8)
{
    const auto bytes = packet (count, frame, points, points);
    Snapshot snapshot;
    EXPECT_TRUE (processor.replayBinaryPosePacketOffline (bytes.data(), (int) bytes.size(), snapshot));
    return snapshot;
}

void samePoint (const DualDLCLiveBridge::PosePoint& a, const DualDLCLiveBridge::PosePoint& b)
{
    EXPECT_EQ (a.valid, b.valid);
    EXPECT_DOUBLE_EQ (a.x, b.x);
    EXPECT_DOUBLE_EQ (a.y, b.y);
    EXPECT_DOUBLE_EQ (a.likelihood, b.likelihood);
}

void sameJsonRoundtripPoint (const DualDLCLiveBridge::PosePoint& binary,
                             const DualDLCLiveBridge::PosePoint& json)
{
    // JSON's decimal serialization can round the promoted double by ~5e-16.
    // Compare this format boundary at the actual DDLP float32 precision.
    // Within-format legacy comparisons below remain exact double comparisons.
    EXPECT_EQ (binary.valid, json.valid);
    EXPECT_EQ ((float) binary.x, (float) json.x);
    EXPECT_EQ ((float) binary.y, (float) json.y);
    EXPECT_EQ ((float) binary.likelihood, (float) json.likelihood);
}

var sideJson (const DualDLCLiveBridge::NativeSideSnapshot& side)
{
    auto* object = new DynamicObject();
    object->setProperty ("frame_id", side.frameId);
    object->setProperty ("picked_side", side.pickedSide);
    object->setProperty ("has_triplet", side.hasTriplet);
    object->setProperty ("has_angle", side.hasAngle);
    object->setProperty ("angle_deg", side.hasAngle ? var (side.angleDeg) : var());
    auto* points = new DynamicObject();
    for (const auto& item : side.points)
    {
        auto* point = new DynamicObject();
        point->setProperty ("valid", item.second.valid);
        point->setProperty ("x", item.second.valid ? var (item.second.x) : var());
        point->setProperty ("y", item.second.valid ? var (item.second.y) : var());
        point->setProperty ("likelihood", item.second.valid ? var (item.second.likelihood) : var());
        points->setProperty (Identifier (String (item.first)), var (point));
    }
    object->setProperty ("points", var (points));
    return var (object);
}

String jsonPacket (int count, int64 frame, const Points& raw)
{
    auto* packet = new DynamicObject();
    packet->setProperty ("schema", "dual_dlc_live.pose.v1");
    packet->setProperty ("pair_index", frame);
    for (const String sideName : { String ("left"), String ("right") })
    {
        auto* side = new DynamicObject();
        side->setProperty ("frame_id", frame);
        auto* points = new DynamicObject();
        for (int i = 0; i < count; ++i)
        {
            auto* point = new DynamicObject();
            point->setProperty ("x", (double) raw[(size_t) i][0]);
            point->setProperty ("y", (double) raw[(size_t) i][1]);
            point->setProperty ("likelihood", (double) raw[(size_t) i][2]);
            points->setProperty (Identifier (String (names[(size_t) i])), var (point));
        }
        side->setProperty ("raw_points", var (points));
        packet->setProperty (sideName, var (side));
    }
    return JSON::toString (var (packet), true);
}
}

TEST (DualDLCLiveOffline, OptionalJsonKneesMatchBinaryWithoutChangingLegacy)
{
    auto binary = makeOfflineProcessor();
    auto json = makeOfflineProcessor();
    auto legacy = makeOfflineProcessor();
    for (int frame = 0; frame < 40; ++frame)
    {
        auto points = samplePoints();
        points[6][0] += float (frame);
        if (frame % 7 == 0) points[6][2] = .01f;
        if (frame == 15) points[6][0] += 1000;
        const auto binarySnapshot = feed (*binary, frame, points);
        Snapshot jsonSnapshot, legacySnapshot;
        ASSERT_TRUE (json->replayJsonPosePacketOffline (jsonPacket (8, frame, points), jsonSnapshot));
        ASSERT_TRUE (legacy->replayJsonPosePacketOffline (jsonPacket (6, frame, points), legacySnapshot));
        EXPECT_EQ (jsonSnapshot.pairIndex, frame);
        EXPECT_EQ (binarySnapshot.ttlWord, jsonSnapshot.ttlWord);
        EXPECT_EQ (jsonSnapshot.ttlWord, legacySnapshot.ttlWord);
        EXPECT_EQ (jsonSnapshot.left.pickedSide, legacySnapshot.left.pickedSide);
        EXPECT_EQ (jsonSnapshot.left.hasAngle, legacySnapshot.left.hasAngle);
        EXPECT_DOUBLE_EQ (jsonSnapshot.left.angleDeg, legacySnapshot.left.angleDeg);
        ASSERT_EQ (jsonSnapshot.left.points.size(), 8u);
        ASSERT_EQ (legacySnapshot.left.points.size(), 6u);
        for (size_t i = 0; i < 8; ++i)
        {
            sameJsonRoundtripPoint (binarySnapshot.left.points.at (names[i]), jsonSnapshot.left.points.at (names[i]));
            sameJsonRoundtripPoint (binarySnapshot.right.points.at (names[i]), jsonSnapshot.right.points.at (names[i]));
            if (i < 6)
                samePoint (legacySnapshot.left.points.at (names[i]), jsonSnapshot.left.points.at (names[i]));
        }
    }
    EXPECT_EQ (json->getCurrentPort(), -1);
    EXPECT_EQ (json->getPendingTtlWordCount(), 0);
}

TEST (DualDLCLiveOffline, SixAndEightHaveIdenticalLegacyOutputs)
{
    auto legacy = makeOfflineProcessor();
    auto knees = makeOfflineProcessor();
    setParam (*legacy, "angle_trigger_enabled", true);
    setParam (*knees, "angle_trigger_enabled", true);
    setParam (*legacy, "angle_threshold_deg", 179.0);
    setParam (*knees, "angle_threshold_deg", 179.0);
    bool observedTriplet = false;
    bool observedAngleTrigger = false;
    for (int frame = 0; frame < 160; ++frame)
    {
        auto points = samplePoints();
        points[0][0] += float (std::sin (frame / 10.0) * 6);
        if (frame % 13 == 0) points[0][2] = .01f;
        if (frame == 35) points[2][0] += 1000;
        if (frame == 55) points[1][0] = std::numeric_limits<float>::quiet_NaN();
        if (frame > 80)
            for (size_t i = 0; i < 6; ++i) points[i][2] = i % 2 ? .95f : .3f;
        points[6] = { float (frame * 1000), -1000, frame % 2 ? .01f : 1.f };
        points[7] = { std::numeric_limits<float>::quiet_NaN(), 0, 1 };
        const auto a = feed (*legacy, frame, points, 6);
        const auto b = feed (*knees, frame, points, 8);
        EXPECT_EQ (a.ttlWord, b.ttlWord);
        observedTriplet |= a.ttlWord != 0;
        observedAngleTrigger |= (a.ttlWord & 12) != 0;
        for (const auto& pair : { std::make_pair (a.left, b.left), std::make_pair (a.right, b.right) })
        {
            EXPECT_EQ (pair.first.hasTriplet, pair.second.hasTriplet);
            EXPECT_EQ (pair.first.hasAngle, pair.second.hasAngle);
            EXPECT_DOUBLE_EQ (pair.first.angleDeg, pair.second.angleDeg);
            EXPECT_EQ (pair.first.pickedSide, pair.second.pickedSide);
            ASSERT_EQ (pair.first.points.size(), 6u);
            ASSERT_EQ (pair.second.points.size(), 8u);
            for (size_t i = 0; i < 6; ++i)
                samePoint (pair.first.points.at (names[i]), pair.second.points.at (names[i]));
        }
    }
    EXPECT_TRUE (observedTriplet);
    EXPECT_TRUE (observedAngleTrigger);
    EXPECT_EQ (legacy->getCurrentPort(), -1);
    EXPECT_EQ (knees->getCurrentPort(), -1);
    EXPECT_EQ (knees->getPendingTtlWordCount(), 0);
    EXPECT_EQ (knees->getPacketsReceived(), 0); // offline is not live acquisition
}

TEST (DualDLCLiveOffline, KneesUseTheIdenticalPointFilter)
{
    auto processor = makeOfflineProcessor();
    setParam (*processor, "enable_hold", true);
    int valid = 0, invalid = 0;
    for (int frame = 0; frame < 80; ++frame)
    {
        auto points = samplePoints();
        points[0][0] += float (frame);
        if (frame == 5) points[0][0] += 1000; // despike
        if (frame >= 10 && frame <= 40) points[0][2] = .01f; // hold then expire
        if (frame == 41) points[0][0] += 800; // gap allows reacquisition
        if (frame == 60) points[0][0] = std::numeric_limits<float>::quiet_NaN();
        points[6] = points[0];
        const auto snapshot = feed (*processor, frame, points);
        const auto& knee = snapshot.left.points.at ("hl_knee_l");
        samePoint (snapshot.left.points.at ("hl_ankle_l"), knee);
        knee.valid ? ++valid : ++invalid;
    }
    EXPECT_GT (valid, 0);
    EXPECT_GT (invalid, 0);
}

TEST (DualDLCLiveOffline, KneeMedianCutoffDespikeGapAndReset)
{
    auto processor = makeOfflineProcessor();
    auto points = samplePoints();
    points[6] = {100, 100, .95f};
    EXPECT_DOUBLE_EQ (feed (*processor, 0, points).left.points.at ("hl_knee_l").x, 100);
    points[6][0] = 110;
    EXPECT_DOUBLE_EQ (feed (*processor, 1, points).left.points.at ("hl_knee_l").x, 105);
    points[6][0] = 120;
    EXPECT_DOUBLE_EQ (feed (*processor, 2, points).left.points.at ("hl_knee_l").x, 110);
    points[6][0] = 900;
    EXPECT_FALSE (feed (*processor, 3, points).left.points.at ("hl_knee_l").valid);
    points[6][2] = .01f;
    EXPECT_FALSE (feed (*processor, 4, points).left.points.at ("hl_knee_l").valid);
    points[6][2] = .95f;
    EXPECT_TRUE (feed (*processor, 20, points).left.points.at ("hl_knee_l").valid);
    points[6][0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE (feed (*processor, 21, points).left.points.at ("hl_knee_l").valid);
    // Existing parameter-change reset must clear added point histories too.
    setParam (*processor, "median_window", 1);
    points[6][0] = 50;
    auto snapshot = feed (*processor, 22, points);
    EXPECT_TRUE (snapshot.left.points.at ("hl_knee_l").valid);
    EXPECT_DOUBLE_EQ (snapshot.left.points.at ("hl_knee_l").x, 50);
    EXPECT_TRUE (snapshot.left.points.at ("hl_knee_r").valid);
}

TEST (DualDLCLiveOffline, KneeHoldUsesSameExpiry)
{
    auto processor = makeOfflineProcessor();
    setParam (*processor, "enable_hold", true);
    setParam (*processor, "max_hold_frames", 2);
    auto points = samplePoints();
    const auto first = feed (*processor, 0, points).left.points.at ("hl_knee_l");
    points[6][2] = .01f;
    const auto held = feed (*processor, 1, points).left.points.at ("hl_knee_l");
    EXPECT_TRUE (held.valid);
    EXPECT_DOUBLE_EQ (held.x, first.x);
    EXPECT_NEAR (held.likelihood, .21, .00001);
    EXPECT_TRUE (feed (*processor, 2, points).left.points.at ("hl_knee_l").valid);
    EXPECT_FALSE (feed (*processor, 3, points).left.points.at ("hl_knee_l").valid);
}

TEST (DualDLCLiveOffline, TruncatedEightDoesNotMutateFilters)
{
    auto processor = makeOfflineProcessor();
    auto points = samplePoints();
    feed (*processor, 0, points);
    points[6][0] = 165; // would shift the median if the rejected packet mutated state
    auto bytes = packet (8, 1, points, points);
    Snapshot snapshot;
    ASSERT_EQ (bytes.size(), 300u);
    EXPECT_FALSE (processor->replayBinaryPosePacketOffline (bytes.data(), (int) bytes.size() - 1, snapshot));
    EXPECT_EQ (processor->getNativePoseSnapshot().pairIndex, 0);
    points[6][0] = 125;
    EXPECT_DOUBLE_EQ (feed (*processor, 2, points).left.points.at ("hl_knee_l").x, 115);
    bytes = packet (7, 3, points, points);
    EXPECT_FALSE (processor->replayBinaryPosePacketOffline (bytes.data(), (int) bytes.size(), snapshot));
}

TEST (DualDLCLiveOffline, ReacquisitionClearsStaleMedianForAllReceivedPoints)
{
    for (int count : {6, 8})
    {
        for (bool despike : {false, true})
        {
            auto processor = makeOfflineProcessor();
            setParam (*processor, "enable_despike", despike);
            const auto base = samplePoints();
            for (int frame = 0; frame < 3; ++frame)
            {
                auto points = base;
                for (auto& point : points) point[0] += float (frame);
                feed (*processor, frame, points, count);
            }
            auto missing = base;
            for (auto& point : missing) point[2] = .01f;
            const auto dropped = feed (*processor, 19, missing, count);
            EXPECT_FALSE (dropped.left.points.at (names[0]).valid);
            auto moved = base;
            for (auto& point : moved) { point[0] += 800; point[1] += 300; }
            const auto acquired = feed (*processor, 20, moved, count);
            for (const auto& side : {acquired.left, acquired.right})
                for (int i = 0; i < count; ++i)
                {
                    const auto& actual = side.points.at (names[(size_t) i]);
                    EXPECT_TRUE (actual.valid);
                    EXPECT_DOUBLE_EQ (actual.x, moved[(size_t) i][0]);
                    EXPECT_DOUBLE_EQ (actual.y, moved[(size_t) i][1]);
                }
            // Smoothing continues normally from the newly acquired segment.
            for (auto& point : moved) point[0] += 4;
            const auto next = feed (*processor, 21, moved, count);
            EXPECT_DOUBLE_EQ (next.left.points.at (names[0]).x, moved[0][0] - 2);
        }
    }
}

TEST (DualDLCLiveOffline, ContinuousMedianAndDespikeBoundaryArePreserved)
{
    auto processor = makeOfflineProcessor();
    const auto base = samplePoints();
    const std::array<float, 7> offsets {{0, 6, 3, 9, 12, 15, 18}};
    const std::array<float, 7> medians {{0, 3, 3, 6, 9, 12, 15}};
    for (size_t frame = 0; frame < offsets.size(); ++frame)
    {
        auto points = base;
        for (auto& point : points) point[0] += offsets[frame];
        const auto snapshot = feed (*processor, (int64) frame, points);
        for (size_t i = 0; i < 8; ++i)
            EXPECT_DOUBLE_EQ (snapshot.left.points.at (names[i]).x, base[i][0] + medians[frame]);
    }
    // A large raw displacement is still rejected at gap == 15; the native
    // despike-reacquisition comparison remains >, independently of median age.
    auto points = base;
    for (auto& point : points) point[0] += 200;
    const auto boundary = feed (*processor, 21, points);
    EXPECT_FALSE (boundary.left.points.at (names[0]).valid);
    // Last accepted frame remains 6, so gap == 16 now permits reacquisition.
    const auto acquired = feed (*processor, 22, points);
    EXPECT_DOUBLE_EQ (acquired.left.points.at (names[0]).x, points[0][0]);
}

TEST (DualDLCLiveOffline, MedianRetainsRecentSampleWithTwoFrameStride)
{
    auto processor = makeOfflineProcessor();
    const auto base = samplePoints();
    for (int step = 0; step < 5; ++step)
    {
        auto points = base;
        for (auto& point : points) point[0] += float (step * 4);
        const auto snapshot = feed (*processor, step * 2, points);
        for (size_t i = 0; i < 8; ++i)
        {
            const double expected = base[i][0] + (step == 0 ? 0 : step * 4 - 2);
            EXPECT_DOUBLE_EQ (snapshot.left.points.at (names[i]).x, expected);
            EXPECT_DOUBLE_EQ (snapshot.right.points.at (names[i]).x, expected);
        }
    }
}

TEST (DualDLCLiveOffline, MedianExpiresSamplesAtThreeFramesOfAge)
{
    auto processor = makeOfflineProcessor();
    auto points = samplePoints();
    const auto base = points;
    feed (*processor, 0, points);
    for (auto& point : points) point[0] += 6;
    feed (*processor, 1, points);
    for (auto& point : points) point[0] += 6;
    const auto partiallyExpired = feed (*processor, 3, points);
    // Frame 0 has age 3 and expires; frame 1 has age 2 and is retained.
    EXPECT_DOUBLE_EQ (partiallyExpired.left.points.at (names[0]).x, base[0][0] + 9);
    for (auto& point : points) point[0] += 6;
    const auto expired = feed (*processor, 6, points);
    for (size_t i = 0; i < 8; ++i)
        EXPECT_DOUBLE_EQ (expired.left.points.at (names[i]).x, points[i][0]);
}

TEST (DualDLCLiveOffline, FrameCounterRestartBeginsFreshMedianSegment)
{
    auto processor = makeOfflineProcessor();
    auto points = samplePoints();
    feed (*processor, 100, points);
    for (auto& point : points) point[0] += 2;
    feed (*processor, 101, points);
    for (auto& point : points) point[0] += 800;
    const auto restarted = feed (*processor, 0, points);
    for (const auto& side : {restarted.left, restarted.right})
        for (size_t i = 0; i < 8; ++i)
        {
            const auto& actual = side.points.at (names[i]);
            EXPECT_TRUE (actual.valid);
            EXPECT_DOUBLE_EQ (actual.x, points[i][0]);
        }
}

TEST (DualDLCLiveOfflineReplay, ReplayFile)
{
    const String inputPath = SystemStats::getEnvironmentVariable ("DDLC_REPLAY_INPUT", "");
    const String outputPath = SystemStats::getEnvironmentVariable ("DDLC_REPLAY_OUTPUT", "");
    if (inputPath.isEmpty() || outputPath.isEmpty())
        GTEST_SKIP() << "Set DDLC_REPLAY_INPUT/OUTPUT for native offline replay";
    auto processor = makeOfflineProcessor();
    const String configPath = SystemStats::getEnvironmentVariable ("DDLC_REPLAY_CONFIG", "");
    if (configPath.isNotEmpty())
    {
        var config = JSON::parse (File (configPath).loadFileAsString());
        ASSERT_TRUE (config.isObject());
        const auto& parameters = config.getDynamicObject()->getProperties();
        for (int i = 0; i < parameters.size(); ++i)
        {
            const String name = parameters.getName (i).toString();
            ASSERT_NE (name, "enabled");
            ASSERT_NE (name, "udp_port");
            setParam (*processor, name, parameters.getValueAt (i));
        }
    }
    FileInputStream input { File (inputPath) };
    FileOutputStream output { File (outputPath) };
    ASSERT_TRUE (input.openedOk());
    ASSERT_TRUE (output.openedOk());
    ASSERT_TRUE (output.setPosition (0));
    ASSERT_TRUE (output.truncate().wasOk());
    int64 records = 0;
    while (! input.isExhausted())
    {
        ASSERT_GE (input.getTotalLength() - input.getPosition(), 4);
        const int size = input.readInt(); // little endian uint32 packet length
        ASSERT_TRUE (size == 252 || size == 300) << size;
        std::vector<char> bytes ((size_t) size);
        ASSERT_EQ (input.read (bytes.data(), size), size);
        Snapshot snapshot;
        ASSERT_TRUE (processor->replayBinaryPosePacketOffline (bytes.data(), size, snapshot)) << records;
        auto* row = new DynamicObject();
        row->setProperty ("pair_index", snapshot.pairIndex);
        row->setProperty ("ttl_word", (int) snapshot.ttlWord);
        row->setProperty ("left", sideJson (snapshot.left));
        row->setProperty ("right", sideJson (snapshot.right));
        const String line = JSON::toString (var (row), true) + "\n";
        ASSERT_TRUE (output.write (line.toRawUTF8(), line.getNumBytesAsUTF8()));
        ++records;
    }
    output.flush();
    EXPECT_TRUE (output.getStatus().wasOk());
    EXPECT_GT (records, 0);
    EXPECT_EQ (processor->getCurrentPort(), -1);
    EXPECT_EQ (processor->getPendingTtlWordCount(), 0);
    RecordProperty ("native_replay_records", std::to_string (records));
}
