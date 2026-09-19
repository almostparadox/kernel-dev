# Linux Kernel Learning Lab

Windows cannot build Linux kernel modules directly. Use WSL2 Ubuntu already installed on system.

## 1. Enter WSL2

Open terminal and run:
```bash
wsl -d Ubuntu
cd /mnt/d/Active-Projects/learn-kernel-dev
```

## 2. Install Build Dependencies (WSL2)

```bash
sudo apt update
sudo apt install -y build-essential linux-headers-generic kmod
```

Note: If WSL kernel lacks matching headers, use QEMU + native Linux kernel build, or install WSL kernel build headers:
```bash
# Optional: build against custom WSL kernel if header mismatch occur
sudo apt install -y flex bison libssl-dev libelf-dev
```

## 3. Projects

- `01-hello-lkm`: Minimal Loadable Kernel Module (`init`, `exit`, `pr_info`).
- `02-char-device`: `miscdevice` driver with user-space read/write and assert test.
- `03-cuda-llm-kernels`: DeepSeek-R1 14B GPU compute kernels (RMSNorm, SwiGLU) with warp shuffle and vectorized memory optimization.
