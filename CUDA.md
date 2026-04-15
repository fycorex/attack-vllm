# CUDA / NVIDIA Driver Runbook

This project is intended to run on an NVIDIA RTX A6000. The current remote
machine state reported by the user is:

- GPU: RTX A6000
- Driver branch: R535
- `nvidia-smi` CUDA driver API: CUDA 12.2
- Symptom: PyTorch warns that the NVIDIA driver is too old and
  `torch.cuda.is_available()` is false or unreliable.

The important distinction:

- The NVIDIA system driver is installed on the host and requires root plus a
  reboot to change.
- The Python/PyTorch CUDA runtime is installed inside `.venv`.
- PyTorch wheels include their own CUDA user-space libraries, but they still
  require a compatible host NVIDIA driver.

## Quick Decision

Use one of these paths:

1. Recommended for A6000: upgrade the host NVIDIA driver to a newer server
   branch, typically R570 or newer, then keep the normal `.venv`.
2. If you cannot reboot or do not have sudo: keep R535 and rebuild `.venv` with
   a PyTorch CUDA 12.1 wheel.

If the installed PyTorch wheel reports CUDA 12.8, R535 is not the right target
driver. NVIDIA's CUDA 12.8 release notes list a toolkit driver requirement of
570.x for CUDA 12.8. R535 is a CUDA 12.2-era driver.

References:

- Ubuntu NVIDIA driver install guide:
  https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/
- NVIDIA Ubuntu data-center driver install guide:
  https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
- NVIDIA CUDA 12.8 release notes:
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html

## Remote Safety First

Do driver work in `tmux`:

```bash
tmux new -s nvidia-driver-fix
```

Before rebooting, make sure you can reconnect to the VM through your cloud or
provider console. Stop experiments before changing the driver:

```bash
nvidia-smi
pkill -f run_caption_attack.py || true
pkill -f run_experiment.sh || true
```

Avoid purging drivers unless you have console access or a reliable snapshot.
A bad driver install can make GPU unavailable until reboot or provider repair.

## Diagnose Current State

Run:

```bash
cat /etc/os-release
uname -r
lspci | grep -i nvidia
nvidia-smi || true
cat /proc/driver/nvidia/version || true
```

Check the Python stack used by this repo:

```bash
cd ~/attack-vllm
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Interpretation:

- `nvidia-smi` shows `Driver Version: 535... CUDA Version: 12.2`: host driver is
  R535 and supports CUDA driver API 12.2.
- `torch.version.cuda` shows `12.8`: your PyTorch wheel expects a newer driver
  than R535 for normal operation.
- `torch.cuda.is_available() == False`: the attack will fall back to CPU and be
  extremely slow.

## Path A: Upgrade The Host Driver

This is the preferred A6000 fix.

Install prerequisite packages:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common build-essential dkms "linux-headers-$(uname -r)"
```

List available server drivers:

```bash
sudo ubuntu-drivers list --gpgpu
```

Prefer an Enterprise Ready Driver package with a `-server` suffix and branch
570 or newer, for example:

```bash
sudo ubuntu-drivers install --gpgpu nvidia:570-server
```

If that exact command is rejected, inspect the list output and install the
available package directly. Examples:

```bash
sudo apt install -y nvidia-driver-570-server
```

or, if only a newer branch is available:

```bash
sudo apt install -y nvidia-driver-575-server
```

Then reboot:

```bash
sudo reboot
```

After reconnecting:

```bash
nvidia-smi
```

Expected:

- Driver version is 570 or newer.
- `nvidia-smi` reports CUDA 12.8 or newer if you installed a new enough branch.

Now verify PyTorch:

```bash
cd ~/attack-vllm
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Expected:

```text
cuda available: True
device: NVIDIA RTX A6000
```

## Path B: Use NVIDIA's CUDA Repository

Use this if Ubuntu's default repositories do not expose a 570+ server driver.

Install NVIDIA's repository keyring:

```bash
cd /tmp
DISTRO=$(. /etc/os-release && echo "ubuntu${VERSION_ID/./}")
wget "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

For A6000, open kernel modules are appropriate because A6000 is Ampere.
NVIDIA's current Ubuntu guide documents:

```bash
sudo apt install -y nvidia-open
```

If that package is unavailable or the machine policy requires proprietary
modules:

```bash
sudo apt install -y cuda-drivers
```

Reboot and verify:

```bash
sudo reboot
```

After reconnect:

```bash
nvidia-smi
cd ~/attack-vllm
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## Path C: Keep R535 And Rebuild `.venv`

Use this only if you cannot update the host driver. It keeps the R535 driver and
uses a PyTorch wheel built for CUDA 12.1.

Python 3.11 is the safest target for older PyTorch CUDA wheels. If the machine
does not have Python 3.11, install it or use the closest supported Python
version available on the host.

```bash
cd ~/attack-vllm
mv .venv .venv-old-cuda128
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

If `pip install -r requirements.txt` upgrades torch to an incompatible wheel,
pin torch explicitly after installing requirements:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then rerun the verification snippet.

## Secure Boot / Kernel Module Troubleshooting

If packages install but `nvidia-smi` fails after reboot:

```bash
mokutil --sb-state || true
dmesg | grep -iE "nvidia|secure|signature|module" | tail -100
lsmod | grep nvidia || true
```

Common causes:

- Secure Boot blocks the NVIDIA kernel module.
- Kernel headers for the running kernel were missing during driver install.
- Old and new NVIDIA packages are mixed.

Install headers and rebuild DKMS:

```bash
sudo apt install -y "linux-headers-$(uname -r)" dkms
sudo dkms autoinstall
sudo reboot
```

If Secure Boot is enabled, either enroll the MOK key during reboot or disable
Secure Boot through the provider console.

## Mixed Package Cleanup

Only do this when you have console access or a provider snapshot:

```bash
dpkg -l | grep -E 'nvidia|cuda' | awk '{print $2, $3}' | sort
```

If packages from multiple driver branches are mixed and the driver cannot load:

```bash
sudo apt purge -y '^nvidia-.*' '^libnvidia-.*' '^cuda-drivers.*'
sudo apt autoremove -y
sudo reboot
```

Then reinstall using Path A or Path B.

## Run The Experiment After Fixing CUDA

Once PyTorch reports CUDA available:

```bash
cd ~/attack-vllm
source .venv/bin/activate
PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_paper.yaml \
  --steps 5 \
  --attack_limit 2
```

If the smoke test works, run the full pipeline:

```bash
bash scripts/run_experiment.sh full
```

## Notes For Containers

If this repo is running inside Docker or another container, update the driver on
the host, not inside the container. Containers see the host driver through the
NVIDIA container runtime. Installing a new driver inside the container will not
fix a host R535 driver mismatch.
