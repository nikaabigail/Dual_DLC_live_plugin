# Сборка и обновление C++ плагина (DualDLCLiveBridge)

Как собрать плагин под целевой Open Ephys и **как обновлять его при правках**.

Исходник плагина — единственный источник истины — здесь: `open_ephys_plugin/DualDLCLiveBridge/`. Чтобы собрать, его нужно положить в дерево **plugin-GUI** (в `Plugins/DualDLCLiveBridge/`) и собрать в нужной конфигурации. Это делает скрипт `scripts/build_plugin.ps1`.

---

## Главное про совместимость (важно прочитать)

Плагин Open Ephys — это DLL, которую грузит GUI. Чтобы она загрузилась, должны совпасть:
1. **Plugin API версия** — тут `10` (и у тебя, и на целевом OE 1.0.1). ✅
2. **Версия GUI / JUCE-ABI** — собирать надо против дерева `plugin-GUI` той же версии, что на целевом (**1.0.1**).
3. **Конфигурация Debug/Release** — на целевом стоит **Release**-инсталлятор Open Ephys. Текущий `dist/windows-x64-debug/DualDLCLiveBridge.dll` — **Debug**, в Release-хост он НЕ загрузится (разный CRT). Нужна **Release**-сборка.
4. **MSVC runtime** — Release-DLL зависит от `vcruntime140.dll`; на целевом должен стоять **VC++ Redistributable x64** (`vc_redist.x64.exe` — это НЕ Visual Studio).

⚠️ **Оговорка про официальный установщик.** На целевом — официальный Release-инсталлятор OE 1.0.1. Наш Release-DLL, собранный против локального дерева `plugin-GUI` 1.0.1, *обычно* грузится, но 100% гарантии нет (официальный релиз мог быть собран чуть другим тулчейном/коммитом). Поэтому: **собрали → скопировали → проверили загрузку**. Если не грузится — см. «Если DLL не грузится».

---

## Сборка (на машине с Visual Studio)

Нужны: Visual Studio с «Desktop development with C++» (у тебя VS 18 Insiders, MSVC 14.51 — ок), CMake, и дерево `plugin-GUI` версии 1.0.1.

```powershell
cd C:\dlc\Dual_DLC_live_plugin
.\scripts\build_plugin.ps1                       # Release; DLL -> dist\windows-x64-release\
# с автодоставкой в папку переноса:
.\scripts\build_plugin.ps1 -DeployDir "D:\transfer\plugins"
# другое дерево GUI:
.\scripts\build_plugin.ps1 -GuiRoot "C:\path\to\plugin-GUI"
```

Скрипт: синкает исходник плагина в `Plugins\DualDLCLiveBridge\` дерева GUI, конфигурит и собирает только цель `DualDLCLiveBridge` в Release, кладёт DLL в `dist\windows-x64-release\` (и в `-DeployDir`, если задан).

> Первая Release-конфигурация собирает зависимости GUI в Release — это долго. Можно собрать и через саму VS: открыть папку `plugin-GUI` как CMake-проект, выбрать конфигурацию **x64-Release**, собрать target **DualDLCLiveBridge**.

## Установка на целевой ПК (без Visual Studio)

1. Скопировать `dist\windows-x64-release\DualDLCLiveBridge.dll` на целевой.
2. Положить в папку плагинов Open Ephys: `%LOCALAPPDATA%\Open Ephys\plugins-api10\` (создать, если нет).
3. Убедиться, что стоит **VC++ Redistributable x64**; если нет — поставить `vc_redist.x64.exe`.
4. Запустить Open Ephys → в signal chain должен появиться узел **`Dual DLCLive Bridge`**.
5. Проверка без камер: `python\send_dual_dlc_bridge_test.py --mode pose --wire-format binary --count 5 --wait-ack` → `acked 5/5`.

## Как обновлять плагин при правках (рабочий цикл)

```
1. правишь open_ephys_plugin\DualDLCLiveBridge\*.cpp / *.h   (в ЭТОМ репо)
2. .\scripts\build_plugin.ps1                                 # пересборка Release -> dist\
3. копируешь dist\windows-x64-release\DualDLCLiveBridge.dll на целевой
   -> %LOCALAPPDATA%\Open Ephys\plugins-api10\  (перезаписать), перезапустить Open Ephys
4. git add -A; git commit -m "plugin: ..."; git push     # фиксируешь исходник (+ DLL) в репо
```

Ключевая мысль: **правишь только в репо** (`open_ephys_plugin\DualDLCLiveBridge\`), скрипт сам синкает в дерево GUI и собирает. Не правь напрямую в `plugin-GUI\Plugins\...` — иначе разъедется источник истины.

Если правок в плагине нет, а поменялся только Python-runtime — пересобирать плагин НЕ нужно (UDP-протокол DDLP/v1 не меняется), достаточно обновить `python\`.

## Если DLL не грузится на целевом

Симптом: узел `Dual DLCLive Bridge` не появляется в списке процессоров.
1. Проверь, что DLL в правильной папке (`plugins-api10`) и что OE её версия = 1.0.1 (API 10).
2. Проверь VC++ Redistributable x64 на целевом.
3. Если всё равно не грузится — это ABI-расхождение с официальным релизом. Варианты:
   - собрать против **официального** `open-ephys/plugin-GUI` на теге **v1.0.1** (а не локального main-дерева), Release;
   - либо перенести на целевой твою **from-source Release-сборку всего GUI** (тогда плагин и хост — из одного дерева, совместимость гарантирована), запускать её `open-ephys.exe` вместо официального;
   - крайний вариант — поставить VS на целевой и собрать там против его GUI.

---

## Тесты плагина

Юнит-тесты живут в `open_ephys_plugin/DualDLCLiveBridge/Tests/` и собираются
целью `DualDLCLiveBridge_tests` внутри дерева plugin-GUI:

```powershell
$G = "C:\path	o\plugin-GUI"
$vc = "C:\Program Files\Microsoft Visual Studio8\Insiders\VC\Auxiliary\Buildcvars64.bat"

# исходники плагина + тесты кладём в дерево GUI
Copy-Item open_ephys_plugin\DualDLCLiveBridge\* "$G\Plugins\DualDLCLiveBridge\" -Recurse -Force

cmd /c "call `"$vc`" && cmake -S `"$G`" -B `"$G\Build-tests`" -G Ninja ^
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DOE_DONT_CHECK_BUILD_PATH=TRUE"
cmd /c "call `"$vc`" && cmake --build `"$G\Build-tests`" --target DualDLCLiveBridge_tests"

# ОБЯЗАТЕЛЬНО: обновить копию DLL, иначе тест гоняет старую
Copy-Item "$G\Build-tests\Plugins\DualDLCLiveBridge.dll" `
          "$G\Build-tests\TestBin\DualDLCLiveBridge\" -Force

& "$G\Build-tests\TestBin\DualDLCLiveBridge\DualDLCLiveBridge_tests.exe"
```

Три вещи, на которых легко споткнуться:

- **Ninja лежит в сборке Visual Studio Insiders**, а не Community. Если брать
  vcvars из Community, CMake не найдёт генератор.
- **`-DOE_DONT_CHECK_BUILD_PATH=TRUE`** обязателен, иначе CMake требует
  конфигурировать строго в папке `Build`.
- **Копировать DLL в `TestBin` руками.** Она попадает туда только при линковке
  exe; если менялся лишь `.cpp` плагина, exe не перелинковывается и тест молча
  проверяет старый бинарь. Проверено мутацией: без копирования убранное гашение
  TTL-линий не ловится вовсе.

Класс помечен макросом `TESTABLE` (как штатные плагины GUI): при `BUILD_TESTS`
он экспортирует символы, иначе разворачивается в пустоту и на боевую сборку не
влияет.
