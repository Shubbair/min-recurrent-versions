# Were RNN all we Needed ?

using the suggested models which outperform the original models like LSTM, GRU and some transformer application , with size and performance.

using min versions of (GRU,LSTM) with the power of parallel , compare it with the original models from Pytorch.

Comparison :

| Model | Original | Minimal |
| --- | --- | --- |
| GRU | ![GRU](assets/GRUV.png) | ![minGRU](assets/GRUV2.png) |
| LSTM | ![LSTM](assets/LSTMV.png) | ![minLSTM](assets/LSTMV2.png) |

See more details in code :

[GRU implementation details](gru_coded_details.ipynb)

[LSTM implementation details](lstm_coded_details.ipynb)

## nn.GRU vs custom GRU

| Aspect | `torch.nn.GRU` (1 layer, uni) | `GRU_Cell` (this repo) | `minGRU_Cell` (this repo) |
| --- | --- | --- | --- |
| API | sequence module | step cell | step cell |
| Call | `gru(x)` | loop: `for t in T: h=cell(x_t,h)` | loop: `for t in T: h=cell(x_t,h)` |
| Params (per hidden unit) | 3 gates: \(3\cdot[(I+H)\cdot H + H]\) | 3 linears on `[x,h]`: \(3\cdot[(I+H)\cdot H + H]\) | 2 linears on `x`: \(2\cdot[I\cdot H + H]\) |
| Gates | reset, update, new | reset, update, new | update only |
| Output | `(out, h_n)` | `h_t` each step | `h_t` each step |

Quick check (shapes, speed):

```python
import torch
from gru_v import GRU_Cell, minGRU_Cell
from test_utils import gru_tests

# Compare PyTorch GRU vs custom step cells
gru_tests(torch.nn.GRU, GRU_Cell)
gru_tests(torch.nn.GRU, minGRU_Cell)
```

## References

[Were RNN Is All We Needed](https://arxiv.org/pdf/2410.01201)
