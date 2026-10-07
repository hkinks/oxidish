"""GPU frame store: current frames only; the previous frame of the stack is the previous row unless `first`."""
import torch


class FrameStore:
    """Ring over time T x N learners (or val windows). Rows are time-major, so row t-1 is the previous decision."""

    def __init__(self, T, N, px=256, device="cuda"):
        self.T, self.N, self.px, self.dev = T, N, px, device
        self.view = torch.zeros((T, N, px, px), dtype=torch.uint8, device=device)
        self.wide = torch.zeros((T, N, px // 2, px // 2), dtype=torch.uint8, device=device)
        self.teacher = torch.zeros((T, N, 4), dtype=torch.float32, device=device)
        self.first = torch.ones((T, N), dtype=torch.bool, device=device)
        self.student = torch.zeros((T, N), dtype=torch.bool, device=device)  # frame reached while the student drove
        self.t = 0       # next write row (absolute)
        self.filled = 0

    def push(self, view0, wide0, teacher, first, student):
        """GPU tensors for N learners (current frame = channel 0)."""
        i = self.t % self.T
        self.view[i], self.wide[i], self.teacher[i] = view0, wide0, teacher
        self.first[i], self.student[i] = first, student
        self.t += 1
        self.filled = min(self.filled + 1, self.T)

    def sample_index(self, B):
        """Uniform over valid (row, learner); the oldest row of a full ring is excluded (its predecessor is gone)."""
        full = self.filled == self.T
        lo = 1 if full else 0
        off = torch.randint(lo, self.filled, (B,), device=self.dev)
        rows = (self.t + off) % self.T if full else off
        cols = torch.randint(0, self.N, (B,), device=self.dev)
        return rows, cols

    def gather(self, rows, cols):
        prev = torch.where(self.first[rows, cols], rows, (rows - 1) % self.T)
        view = torch.stack([self.view[rows, cols], self.view[prev, cols]], 1)
        wide = torch.stack([self.wide[rows, cols], self.wide[prev, cols]], 1)
        return view, wide, self.teacher[rows, cols], self.student[rows, cols]
