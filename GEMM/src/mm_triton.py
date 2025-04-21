import triton
import triton.language as tl
import torch

# Define matrix dimensions
A_ROW = 640
A_COLUMN = 12800
B_ROW = 12800
B_COLUMN = 640
C_ROW = 640
C_COLUMN = 640


# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    # Compute row and column indices
    row = pid // (N // BLOCK_SIZE) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = pid % (N // BLOCK_SIZE) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE):
        # Load A and B tiles
        a = tl.load(
            A_ptr
            + (row[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE)) * stride_ak),
            mask=row[:, None] < M,
        )
        b = tl.load(
            B_ptr
            + (
                (k + tl.arange(0, BLOCK_SIZE))[:, None] * stride_bk
                + col[None, :] * stride_bn
            ),
            mask=col[None, :] < N,
        )
        # Compute partial product
        acc += tl.dot(a, b)
    # Write back to C
    c = acc.to(tl.float32)
    tl.store(
        C_ptr + (row[:, None] * stride_cm + col[None, :] * stride_cn),
        c,
        mask=(row[:, None] < M) & (col[None, :] < N),
    )


# Wrapper function for Triton kernel
def matmul(A, B):
    assert A.shape[1] == B.shape[0], "Matrix dimensions do not match for multiplication"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE"]) * triton.cdiv(N, META["BLOCK_SIZE"]),
    )
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE=32,
    )
    return C


# Example usage
if __name__ == "__main__":
    # Initialize matrices
    A = torch.randn(A_ROW, A_COLUMN, device="cuda", dtype=torch.float32)
    B = torch.randn(B_ROW, B_COLUMN, device="cuda", dtype=torch.float32)
    # Perform matrix multiplication
    C = matmul(A, B)
    # Verify result
    torch_result = torch.matmul(A, B)
    print("Difference:", torch.norm(C - torch_result).item())
