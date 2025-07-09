# A benchmark with triton to compare with raw CUDA and native pytorch
# It is observed that for big H and W values, Triton performs a lot worse than native torch
# For instance, H=W=512

# For H=W=128, Triton is superior
import torch
import triton
import triton.language as tl
import time

# -----------------------------------------------------------------------------
# Triton kernel
# -----------------------------------------------------------------------------
@triton.jit
def conv3x3_bias_relu(
    # pointers
    X_ptr, K_ptr, b_ptr, Y_ptr,
    # tensor dims
    B, C_in, C_out, H, W,
    # strides
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_kcout, stride_kcin, stride_ki, stride_kj,
    stride_yb, stride_ycout, stride_yh, stride_yw,
    # tiling
    BLOCK_HW: tl.constexpr, 
    BLOCK_Cout: tl.constexpr,
):
    # each program ID handles one tile of size BLOCK_HW × BLOCK_Cout
    pid = tl.program_id(0)
    cout_blk = tl.program_id(1)
    b   = tl.program_id(2)

    # offsets in output
    offs_hw = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h = offs_hw // W
    w = offs_hw % W
    couts = cout_blk * BLOCK_Cout + tl.arange(0, BLOCK_Cout)

    mask_hw = (offs_hw < H * W)
    mask_cout = (couts < C_out)

    # load bias
    bias = tl.load(b_ptr + couts, mask=mask_cout, other=0.0)

    # init result with bias
    acc = tl.zeros((BLOCK_Cout, BLOCK_HW), dtype=tl.float32)
    acc += bias[:, None]

    # for each cin and 3×3 kernel
    for cin in range(0, C_in):
        for i in range(3):
            for j in range(3):
                # compute input coords
                hh = h + i - 1
                ww = w + j - 1
                mask_in = mask_hw & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
                # ptrs
                x_ptrs = X_ptr + b*stride_xb + cin*stride_xc + hh*stride_xh + ww*stride_xw
                k_ptrs = K_ptr + couts*stride_kcout + cin*stride_kcin + i*stride_ki + j*stride_kj
                # load
                x = tl.load(x_ptrs, mask=mask_in, other=0.0)
                k = tl.load(k_ptrs, mask=mask_cout, other=0.0)
                # broadcast x to [Cout, HW] and k to same shape
                acc += k[:, None] * x[None, :]

    # ReLU
    acc = tl.where(acc > 0, acc, 0.0)

    # write back
    y_ptrs = Y_ptr + b*stride_yb + couts[:, None]*stride_ycout + h[None, :]*stride_yh + w[None, :]*stride_yw
    tl.store(y_ptrs, acc, mask=mask_hw[None, :] & mask_cout[:, None])

# -----------------------------------------------------------------------------
# Driver / benchmarking
# -----------------------------------------------------------------------------
def benchmark():
    torch.manual_seed(0)
    dtype = torch.float32
    device = 'cuda'

    # example dims
    B, C_in, C_out, H, W = 8, 32, 64, 512, 512
    X = torch.randn(B, C_in, H, W, device=device, dtype=dtype)
    K = torch.randn(C_out, C_in, 3, 3, device=device, dtype=dtype)
    b = torch.randn(C_out, device=device, dtype=dtype)
    Y = torch.empty(B, C_out, H, W, device=device, dtype=dtype)

    # strides (dimension width essentially to help calculate offset)
    sx = X.stride()
    sk = K.stride()
    sy = Y.stride()

    # tiling parameters
    BLOCK_HW = 256
    BLOCK_Cout = 16

    # warm-up
    for _ in range(5):
        conv3x3_bias_relu[( (H*W+BLOCK_HW-1)//BLOCK_HW, 
                             (C_out+BLOCK_Cout-1)//BLOCK_Cout, 
                             B )](
            X, K, b, Y,
            B, C_in, C_out, H, W,
            sx[0], sx[1], sx[2], sx[3],
            sk[0], sk[1], sk[2], sk[3],
            sy[0], sy[1], sy[2], sy[3],
            BLOCK_HW, BLOCK_Cout
        )
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(30):
        conv3x3_bias_relu[( (H*W+BLOCK_HW-1)//BLOCK_HW,
                             (C_out+BLOCK_Cout-1)//BLOCK_Cout,
                             B )](
            X, K, b, Y,
            B, C_in, C_out, H, W,
            sx[0], sx[1], sx[2], sx[3],
            sk[0], sk[1], sk[2], sk[3],
            sy[0], sy[1], sy[2], sy[3],
            BLOCK_HW, BLOCK_Cout
        )
    torch.cuda.synchronize()
    t_triton = (time.time() - t0)/30*1000
    print(f"Triton kernel: {t_triton:.2f} ms")

    # compare with PyTorch native conv2d + bias + ReLU
    t0 = time.time()
    for _ in range(30):
        Y2 = torch.nn.functional.conv2d(X, K, bias=b, padding=1)
        Y2 = torch.relu(Y2)
    torch.cuda.synchronize()
    t_pt = (time.time() - t0)/30*1000
    print(f"PyTorch fused: {t_pt:.2f} ms")

if __name__ == '__main__':
    benchmark()
