/*
    Тесты DualDLCLiveBridge. Покрыт сторож по возрасту пакета.

    ЗАЧЕМ ИМЕННО ОН. queueTtlWord эмитит только при ИЗМЕНЕНИИ TTL-слова. Если
    источник (Python) упал, завис или потерял камеру в момент, когда линия
    поднята, слово менять больше некому - и линия останется HIGH бесконечно.
    В замкнутом контуре это залипший стимуляционный гейт на живом животном,
    причём внешне всё выглядит штатно.

    ЧТО ПРОВЕРЯЕТСЯ. Решение сторожа на настоящем сокете: посылаем пакет,
    поднимающий линию, убеждаемся что слово поднялось, замолкаем - и проверяем,
    что сторож сам сбросил слово в ноль.

    ЧЕГО ТЕСТ НЕ ПОКРЫВАЕТ И ПОЧЕМУ. Не доходит до выдачи TTL-события в поток:
    в тестовом харнесе GUI setTTLState разыменовывает ttlEventChannel, который
    в харнесе не проставляется, и любой process() с непустой очередью падает с
    access violation. Это ограничение харнеса, а не плагина: проверено
    отдельными диагностическими тестами, что создание процессора, чтение
    параметров, открытие сокета, приём пакета и process() с ПУСТОЙ очередью
    проходят штатно. Поэтому сторож вызывается напрямую, а сквозная проверка
    "фронт дошёл до Open Ephys" остаётся ручной, по процедуре в
    docs/RUN_EXPERIMENT.md.
    КАК ЗАПУСКАТЬ (см. docs/BUILD_PLUGIN.md, раздел про тесты).

    ВАЖНО: после сборки DLL плагина надо СКОПИРОВАТЬ её в TestBin руками.
    В TestBin она попадает только при ЛИНКОВКЕ exe, а если менялся лишь .cpp
    плагина, exe не перелинковывается - и тест молча гоняет СТАРУЮ библиотеку.
    На этом легко получить зелёный прогон по правке, которой в бинаре нет:
    проверено мутацией, без ручного копирования убранное гашение линий не
    ловилось вовсе, а с копированием два теста падают как надо.
*/
#include <chrono>
#include <thread>

#include "gtest/gtest.h"

#include "../DualDLCLiveBridge.h"
#include <ModelApplication.h>
#include <ModelProcessors.h>
#include <ProcessorHeaders.h>
#include <TestFixtures.h>

namespace
{
constexpr int kWatchdogMs = 100;
}

class DualDLCLiveBridgeTests : public testing::Test
{
protected:
    void SetUp() override
    {
        tester = std::make_unique<ProcessorTester> (TestSourceNodeBuilder (FakeSourceNodeParams {
            1,
            30000.0f,
            1.0,
        }));
        processor = tester->createProcessor<DualDLCLiveBridge> (Plugin::Processor::UTILITY);
        ASSERT_NE (processor, nullptr);
        ASSERT_GT (processor->getCurrentPort(), 0) << "сокет не открылся";
    }

    /*  Параметр надо и записать, и уведомить процессор: currentValue сам по
        себе parameterValueChanged не вызывает, а таймаут кэшируется там. */
    void setWatchdogMs (int ms)
    {
        auto* param = processor->getParameter ("watchdog_timeout_ms");
        ASSERT_NE (param, nullptr);
        param->currentValue = ms;
        processor->parameterValueChanged (param);
    }

    /*  Пакет legacy-протокола, поднимающий одну линию. Ждём, пока сокетный
        поток плагина его заберёт, иначе тест гоняет старое состояние. */
    void sendLineAndWait (int line)
    {
        const int64 before = processor->getPacketsReceived();

        String lines;
        for (int i = 0; i < 8; i++)
            lines += String (i == line ? "true" : "false") + (i < 7 ? "," : "");
        const String json = "{\"schema\":\"dual_dlc_live.v1\",\"pair_index\":1,"
                            "\"ttl_lines\":["
                            + lines + "]}";

        DatagramSocket sender;
        ASSERT_GT (sender.write ("127.0.0.1",
                                 processor->getCurrentPort(),
                                 json.toRawUTF8(),
                                 (int) json.getNumBytesAsUTF8()),
                   0)
            << "пакет не отправился";

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds (2);
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (processor->getPacketsReceived() > before)
                return;
            std::this_thread::sleep_for (std::chrono::milliseconds (5));
        }
        FAIL() << "плагин не принял пакет";
    }

    static void goSilent (int ms) { std::this_thread::sleep_for (std::chrono::milliseconds (ms)); }

    std::unique_ptr<ProcessorTester> tester;
    DualDLCLiveBridge* processor = nullptr;
};

/*  Основной сценарий: линия поднята, источник замолчал, слово обязано обнулиться. */
TEST_F (DualDLCLiveBridgeTests, WatchdogClearsTtlWhenSourceGoesSilent)
{
    setWatchdogMs (kWatchdogMs);
    sendLineAndWait (4);

    ASSERT_EQ (processor->getLastTtlWord(), 1 << 4) << "линия не поднялась";
    processor->applyPacketWatchdog();
    EXPECT_FALSE (processor->isWatchdogTripped()) << "сработал, хотя пакет свежий";

    goSilent (kWatchdogMs * 3);
    processor->applyPacketWatchdog();

    EXPECT_EQ (processor->getLastTtlWord(), 0) << "линия залипла поднятой";
    EXPECT_TRUE (processor->isWatchdogTripped());
    EXPECT_EQ (processor->getWatchdogTrips(), 1);
}

/*  Пока пакеты идут, сторож молчит. */
TEST_F (DualDLCLiveBridgeTests, WatchdogStaysQuietWhilePacketsArrive)
{
    setWatchdogMs (kWatchdogMs);

    for (int i = 0; i < 5; i++)
    {
        sendLineAndWait (4);
        processor->applyPacketWatchdog();
        goSilent (kWatchdogMs / 4);
    }

    EXPECT_EQ (processor->getLastTtlWord(), 1 << 4);
    EXPECT_FALSE (processor->isWatchdogTripped());
    EXPECT_EQ (processor->getWatchdogTrips(), 0);
}

/*  Нулевой таймаут выключает сторожа. Это же проверка, что тест выше ловит
    именно сторожа, а не какой-то посторонний сброс. */
TEST_F (DualDLCLiveBridgeTests, WatchdogDisabledLeavesTtlUntouched)
{
    setWatchdogMs (0);
    sendLineAndWait (4);
    ASSERT_EQ (processor->getLastTtlWord(), 1 << 4);

    goSilent (kWatchdogMs * 3);
    processor->applyPacketWatchdog();

    EXPECT_EQ (processor->getLastTtlWord(), 1 << 4) << "сработал, хотя выключен";
    EXPECT_FALSE (processor->isWatchdogTripped());
    EXPECT_EQ (processor->getWatchdogTrips(), 0);
}

/*  После возвращения источника сторож должен взводиться заново, иначе он
    отработает ровно один раз за сессию и дальше будет бесполезен. */
TEST_F (DualDLCLiveBridgeTests, WatchdogRearmsAfterSourceRecovers)
{
    setWatchdogMs (kWatchdogMs);

    sendLineAndWait (4);
    goSilent (kWatchdogMs * 3);
    processor->applyPacketWatchdog();
    ASSERT_TRUE (processor->isWatchdogTripped());
    ASSERT_EQ (processor->getWatchdogTrips(), 1);

    sendLineAndWait (4);                       // источник вернулся
    processor->applyPacketWatchdog();
    EXPECT_FALSE (processor->isWatchdogTripped()) << "взвод не снялся";
    EXPECT_EQ (processor->getLastTtlWord(), 1 << 4);

    goSilent (kWatchdogMs * 3);                // и снова замолчал
    processor->applyPacketWatchdog();
    EXPECT_EQ (processor->getLastTtlWord(), 0);
    EXPECT_EQ (processor->getWatchdogTrips(), 2) << "не сработал повторно";
}

/*  Пока пакетов не было вовсе, сторожу срабатывать не на чем. */
TEST_F (DualDLCLiveBridgeTests, WatchdogIgnoresSessionBeforeFirstPacket)
{
    setWatchdogMs (kWatchdogMs);
    goSilent (kWatchdogMs * 3);
    processor->applyPacketWatchdog();

    EXPECT_FALSE (processor->isWatchdogTripped());
    EXPECT_EQ (processor->getWatchdogTrips(), 0);
}
