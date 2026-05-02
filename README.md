# picam

Real-time object detection on a Raspberry Pi 5 using the Sony IMX500 AI Camera, displayed on a 5" portrait touchscreen.

---

## Parts List

| Part | Details |
|------|---------|
| [Raspberry Pi 5 Starter Kit](https://thepihut.com/products/raspberry-pi-5-starter-kit) | Pi 5, 32GB SD card, 27W USB-C PSU, official case |
| [Raspberry Pi AI Camera](https://thepihut.com/products/raspberry-pi-ai-camera) | Sony IMX500, 12.3MP, on-chip AI inference |
| [Raspberry Pi Touch Display 2](https://thepihut.com/products/raspberry-pi-touch-display-2) | 5", 720×1280, DSI, capacitive touch |
| [Raspberry Pi Active Cooler](https://thepihut.com/products/raspberry-pi-active-cooler) | Fan + heatsink for Pi 5 |
| Logitech K400+ (optional) | Wireless keyboard with trackpad for local input |

---

## Hardware Installation

### 1. Active Cooler

1. Remove the cooler from its packaging — do not peel the thermal pad.
2. Align the cooler over the Pi 5 CPU (the large silver chip in the centre of the board).
3. Press the two plastic push-pins into the mounting holes beside the CPU until they click.
4. Connect the fan's JST connector to the FAN port on the Pi 5 (near the USB-A ports).

> The cooler does not fit with the official case lid when the Touch Display 2 is mounted — leave the lid off.

### 2. Touch Display 2

1. Lay the display face-down and locate the DSI ribbon cable connector on its back.
2. Connect the ribbon cable to the **DISP0** (or **CAM/DISP0**) connector on the Pi 5. Lift the retaining latch, insert the cable blue-side up, then press the latch down.
3. Connect the display's 5V power cable to the Pi 5 GPIO header — **Pin 2 (5V)** and **Pin 6 (GND)**.
4. After booting, enable the display driver — see [Environment Setup](#environment-setup).

### 3. AI Camera

1. Locate the **CAM1** (or **CAM/DISP1**) connector on the Pi 5 — the one not used by the display.
2. Lift the retaining latch on the connector, insert the camera's ribbon cable with the contacts facing the board, then press the latch down.
3. Verify detection after booting:
   ```bash
   rpicam-hello --list-cameras
   # Should show: imx500 [4056x3040]
   ```

---

## Environment Setup

### 1. Flash Pi OS

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash **Raspberry Pi OS (64-bit)** (Debian Trixie) to the SD card.

In the imager's OS Customisation settings:
- Set hostname (e.g. `picam`)
- Set username and password
- Enable SSH (password authentication)
- Configure Wi-Fi if needed

> **Headless Wi-Fi gotcha:** Pi OS ships with Wi-Fi rfkill soft-blocked. If the Pi doesn't connect headlessly, run `sudo rfkill unblock wifi` from a local terminal session or write `0` to `/var/lib/systemd/rfkill/platform-1001100000.mmc:wlan` before booting. It is easier to do initial setup with a monitor and keyboard.

### 2. First Boot

Connect a monitor, keyboard, and mouse for first boot. Complete the setup wizard, then connect to your network.

Update the system:
```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Enable Desktop Auto-Login

The app uses OpenCV's `imshow` which requires a running display server. Enable desktop auto-login so the GUI session starts on boot:

```bash
sudo raspi-config nonint do_boot_behaviour B4
sudo reboot
```

### 4. Enable the Touchscreen

Add the display overlay to `/boot/firmware/config.txt`:

```bash
echo "dtoverlay=vc4-kms-dsi-ili9881-5inch" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

After reboot the display should show the desktop in portrait orientation.

### 5. SSH Key Setup (optional)

To use key-based SSH from your development machine:

```bash
# On your dev machine
ssh-copy-id -i ~/.ssh/your_key.pub devrim@<pi-ip>
```

Connect with:
```bash
ssh -i ~/.ssh/your_key -o IdentitiesOnly=yes devrim@<pi-ip>
```

### 6. Python Virtual Environment

`picamera2` is installed via apt on Pi OS. The venv needs `--system-site-packages` to access it:

```bash
cd ~/picam
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

---

## Software

### Architecture

```
Sony IMX500 (on-camera inference)
        │ CSI (tensor output)
        ▼
    picamera2 / libcamera
        │
    modlib AiCamera
        │  frame.detections (parsed, normalised)
        ▼
    main.py
        │  rotate 90° CW → resize to 720×1280
        ▼
    OpenCV imshow (fullscreen on Touch Display 2)
```

The IMX500 runs the neural network on-chip and sends detection tensors alongside the image over CSI. `modlib` wraps `picamera2` and handles all tensor postprocessing, giving clean `frame.detections` objects without manual tensor parsing.

### Dependencies

| Package | Purpose |
|---------|---------|
| `modlib` | Sony's Application Module Library — IMX500 abstraction, model zoo, detection parsing |
| `opencv-python` | Frame display and image transforms |
| `picamera2` | Installed via apt (`sudo apt install python3-picamera2`) |

### Model

`SSDMobileNetV2FPNLite320x320` from the modlib model zoo — a COCO-trained single-shot detector. The `.rpk` network firmware is downloaded to `~/.modlib/zoo/` on first run and uploaded to the IMX500 at startup (takes ~5 seconds).

Input: 320×320. Output: up to 100 detections per frame (boxes, scores, class IDs). COCO 80-class labels.

### Running

```bash
cd ~/picam
DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/1000 .venv/bin/python main.py
```

> Pi OS Trixie uses Wayland (labwc). All three environment variables are required when launching a GUI app over SSH.

Press `q` to quit.

### Display

The Touch Display 2 is portrait (720×1280). The camera captures landscape at 640×480. Each frame is:
1. Annotated with bounding boxes and labels by `modlib`'s `Annotator`
2. Rotated 90° clockwise with `cv2.ROTATE_90_CLOCKWISE`
3. Resized to 720×1280 and shown fullscreen

### Frame Rate

`AiCamera(frame_rate=15)` matches the camera capture rate to the IMX500's detection rate (~15 DPS). Running at 30fps with 15 DPS causes every other frame to have no detections, making boxes flicker.

### Development Workflow

The repo lives at `github.com/dersanli/picam`. To deploy changes to the Pi:

```bash
# On dev machine
git push origin main

# On Pi (or via SSH)
cd ~/picam && git pull
```
