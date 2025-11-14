# Fist time: from the project root

```
source .venv/bin/activate
python -m pip install -r requirements.txt
```
# Afterwards entering the virtual environment
```
source .venv/bin/activate
```
# compiling the app
```
python -m compileall balance_game                                                
```
# run the app with 2 terminal
```
source .venv/bin/activate
python main.py --socket-input --socket-port 4789
```

```
source .venv/bin/activate
python tools/neuroskypy_socket_bridge.py \
    --serial-port /dev/cu.BrainLink_Lite \
    --profile assets/blink_energy_profile.json \
    --game-port 4789
```



## Blink-to-jump via BrainLink

The game can react to BrainLink / NeuroSky blink events using the built-in ThinkGear socket service.

1. Pair and start the BrainLink headset with the official ThinkGear Connector (or compatible service).  
   Ensure it is streaming JSON packets on `127.0.0.1:13854`.
2. Launch the game with blink support:

   ```bash
   python main.py --brainlink
   ```

Optional overrides:

- `--blink-threshold <value>` – change the blink strength needed to trigger a jump (default 55).
- `--brainlink-host <host>` / `--brainlink-port <port>` – connect to a non-default ThinkGear socket.

You can still lean/jump via keyboard; successful blinks act like tapping the jump key.

## External control via JSON socket

If your ML model or AutoHotKey script already interprets BrainLink data, you can stream the
resulting control signals straight into the game.

1. Start the game with the socket listener enabled (defaults to `127.0.0.1:4789`):

   ```bash
   python main.py --socket-input
   ```

   Use `--socket-host` / `--socket-port` to change the bind address.

2. From your pipeline, open a TCP connection to that address and send newline-delimited JSON
   messages such as:

   ```json
   {"lean": -0.35}
   {"jump": true}
   {"jump": false}
   ```

   - `lean` accepts values between `-1.0` (hard left) and `1.0` (hard right).
   - `jump` acts like pressing and releasing the jump key; short pulses are enough.
   - Include both fields in one message if you prefer: `{"lean": 0.1, "jump": true}`.
   - Optional `{"reset": true}` returns control to the keyboard baseline.

The socket layer stacks with the keyboard and blink input, so you can fall back to manual control at any time.

## Blink energy training + BrainLink bridge

1. **Derive an energy profile（一次即可）**

   ```
   python tools/train_blink_energy.py \
       --datasets ~/Downloads/BME_Lab_BCI_training/bci_dataset_114-1 \
                 ~/Downloads/BME_Lab_BCI_training/bci_dataset_113-2 \
       --output assets/blink_energy_profile.json
   ```

   這會讀取各受試者的 `S*/3.txt`（含 20 秒睜眼／20 秒閉眼循環），計算開眼與閉眼的能量分佈並輸出
   建議的能量閾值。結果會寫進 `assets/blink_energy_profile.json`，後續橋接程式與即時偵測會自動讀取。

2. **啟動遊戲的 socket listener**

   ```
   python main.py --socket-input
   ```

3. **執行 BrainLink → 模型 → 遊戲的橋接腳本**

   ```
   python tools/brainlink_socket_bridge.py \
       --thinkgear-host 127.0.0.1 --thinkgear-port 13854 \
       --game-port 4789 \
       --profile assets/blink_energy_profile.json \
       --model-module your_ml_module
   ```

   - `--profile` 指向上一部產生的能量設定，會驅動 `EnergyBlinkDetector` 讀取 raw EEG（需先開啟 ThinkGear Connector）。
   - `--model-module` 是選填的 Python 模組，需提供 `predict(packet: dict) -> dict`，可以在裡面載入同學的專注/放鬆模型並輸出
     `{"lean": …, "jump": …}`。若未指定，預設用冥想值對應傾斜，眨眼則由能量檢測決定。
   - 若你的模型也要外送 JSON，可直接在 `predict` 回傳字典即可。

4. 桥接腳本會把每次眨眼（能量短暫下降）轉成 `{"jump": true}` 的 JSON 指令送進遊戲的 socket。
   你也可以在自訂模組中利用 `packet["rawEeg"]` 自行處理特徵。

## 直接用 NeuroSkyPy 連 BrainLink（不用 ThinkGear Connector）

1. 安裝需求（只需一次）：
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 查詢 BrainLink/MindWave 的藍牙序列埠名稱（macOS 通常是 `/dev/tty.MindWaveMobile-SerialPort`）。
3. 啟動遊戲 socket：
   ```bash
   python main.py --socket-input
   ```
4. 使用我們提供的橋接程式，直接透過 NeuroSkyPy 讀取 BrainLink：
   ```bash
   python tools/neuroskypy_socket_bridge.py \
       --serial-port /dev/tty.MindWaveMobile-SerialPort \
       --profile assets/blink_energy_profile.json \
       --game-port 4789 \
       --model-module your_ml_module   # 若沒有可省略
   ```

   - 腦波 raw 資料會直接經 `EnergyBlinkDetector` 做能量尖峰偵測 → 觸發 jump。
   - 如果提供 `--model-module`，請在該模組內定義 `predict(packet: dict) -> dict`，回傳 `{"lean": …}` 等欄位即可。
   - 沒有能量 profile 時，會 fallback 成以 `blinkStrength` 閾值判斷眨眼。

> 注意：NeuroSkyPy 需要正確的藍牙 Serial Port，且腳本關閉時會自動釋放連線；若要結束，按 `Ctrl+C` 即可。
