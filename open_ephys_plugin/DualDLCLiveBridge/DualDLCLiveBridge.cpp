/*
    ------------------------------------------------------------------

    Dual DLCLive bridge for Open Ephys GUI.

    Receives compact UDP JSON packets from dual_rt_dlc_live.py and mirrors
    the packet's ttl_lines array into an Open Ephys TTL event channel.

    ------------------------------------------------------------------
*/

#include "DualDLCLiveBridge.h"

namespace
{
constexpr size_t maxPendingTtlWords = 512;
}

DualDLCLiveBridge::DualDLCLiveBridge()
    : GenericProcessor ("Dual DLCLive Bridge"),
      Thread ("Dual DLCLive Bridge UDP")
{
    emittedLineStates.fill (false);
    for (auto& line : desiredLineStates)
        line.store (false);
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
    if (param->getName().equalsIgnoreCase ("enabled")
        || param->getName().equalsIgnoreCase ("udp_port"))
    {
        ensureSocket();
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
    currentPort.store (-1);
}

void DualDLCLiveBridge::run()
{
    constexpr int maxPacketsPerWake = 64;
    constexpr int maxPacketBytes = 8192;
    char buffer[maxPacketBytes] {};

    while (! threadShouldExit())
    {
        for (int packet = 0; packet < maxPacketsPerWake && ! threadShouldExit(); packet++)
        {
            String message;
            String senderAddress;
            int senderPort = 0;

            {
                const ScopedLock lock (socketLock);
                if (socket == nullptr)
                    break;

                const int timeoutMs = packet == 0 ? 100 : 0;
                if (socket->waitUntilReady (true, timeoutMs) <= 0)
                    break;

                const int bytesRead = socket->read (buffer, maxPacketBytes - 1, false, senderAddress, senderPort);
                if (bytesRead <= 0)
                    break;

                buffer[bytesRead] = 0;
                message = String::fromUTF8 (buffer, bytesRead).trim();
            }

            if (message.isNotEmpty())
            {
                String ackMessage;
                if (applyMessage (message, ackMessage) && ackMessage.isNotEmpty() && senderPort > 0)
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
    if (schema.isNotEmpty() && schema != "dual_dlc_live.v1")
        return false;

    const var ttlLines = parsed.getProperty ("ttl_lines", var());
    if (! ttlLines.isArray())
        return false;

    Array<var>* lines = ttlLines.getArray();
    const int nLines = jmin (8, lines->size());
    std::array<bool, 8> nextStates {};
    for (int line = 0; line < 8; line++)
        nextStates[(size_t) line] = desiredLineStates[(size_t) line].load();

    for (int line = 0; line < nLines; line++)
    {
        nextStates[(size_t) line] = bool (lines->getReference (line));
    }

    uint8 ttlWord = 0;
    for (int line = 0; line < 8; line++)
    {
        desiredLineStates[(size_t) line].store (nextStates[(size_t) line]);
        if (nextStates[(size_t) line])
            ttlWord |= static_cast<uint8> (1 << line);
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

        ackMessage = "dual_dlc_live.ack pair=" + String (pairIndex) + " ttl=0x" + ttlHex + "\n";
    }
    return true;
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
