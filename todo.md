# HRM Blackwell GPU Setup Troubleshooting & Tutorial Plan

## Objectives
1. **Fix Blackwell GPU compatibility issues:**
   - Install PyTorch 2.7.0 with CUDA 12.8 support for Blackwell GPUs (sm_120/sm_122)
   - Enable full NVIDIA Blackwell GPU support (RTX 50-series, B200, RTX Pro 6000)
   - Troubleshoot and resolve CUDA compatibility warnings

2. **Create comprehensive setup tutorial once working**

3. **Recommend future work**

## Current Status - BREAKTHROUGH: Blackwell GPU RECOGNIZED!
- Virtual environment: ✅ Created
- PyTorch: ⚠️ 2.10.0.dev20251121+cu128 - GPU recognized but Blackwell CC 12.1 exceeds max supported CC 12.0
- FlashAttention: ❌ Not tried yet - need CC 12.1 support in PyTorch first
- Dependencies: ✅ Core packages installed
- GPU: NVIDIA GB10 (Blackwell, CUDA CC 12.1) - ✅ DETECTED by NVIDIA ML and PyTorch!
- Driver: 580.95.05 (CUDA 13.0) - ✅ Compatible
- **Root Cause Found**: Blackwell compute capability 12.1 > PyTorch's max supported 12.0
- **Next Step**: Find PyTorch build supporting compute capability 12.1+

## Troubleshooting Steps Needed

### 1. Install PyTorch 2.7.0 with Blackwell Support
```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```
*Note: This provides pre-built wheels for Blackwell GPUs with CUDA 12.8*

### 2. Verify Blackwell GPU Recognition
- Test `torch.cuda.is_available()`
- Check `torch.cuda.get_device_name(0)`
- Ensure `torch.version.cuda == "12.8"`

### 3. Install FlashAttention (for Hopper/Blackwell)
- Build Flash-Attention 3 for Blackwell GPUs
- Verify compatibility with PyTorch 2.7.x

### 4. Install Python Dependencies
- Resolve networkx version conflicts
- Install all packages without version errors

### 5. Test HRM Model Instantiation
- Import HRM modules
- Create basic model
- Run forward pass on CPU (then GPU if available)

### 6. Test Training Pipeline
- Run small scale training on CPU
- Test data loading and processing

## Timeline
- **Week 1-2**: PyTorch 2.7.0 Blackwell support becomes stable (April-May 2025)
- **Immediate**: Use nightly builds if needed
- **Current**: Update to 2.7 stable release

## Files to Create
- `tutorial.md`: Complete setup documentation
- Update this file with progress and bugs encountered

## Known Issues
1. Blackwell GPUs are very new (released 2025), limited software support
2. PyTorch nightly builds may be needed initially
3. Windows Blackwell support may be limited compared to Linux
4. Some dependencies may need manual version pinning

## Success Criteria
- ✅ `torch.cuda.get_device_name(0)` returns "NVIDIA GB10" without warnings
- ✅ HRM model can be instantiated and run forward pass on GPU
- ✅ Basic training loop completes without errors
- ✅ All dependencies installed cleanly
