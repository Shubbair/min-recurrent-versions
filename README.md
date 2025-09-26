# Were RNN all we Needed ?

using the suggested models which outperform the original models like LSTM, GRU and some transformer application , with size and performance.

using min versions of (GRU,LSTM) with the power of parallel , compare it with the original models from Pytorch.

## Comparison

**minGRU**

![minGRU](assets/GRUV2.png)

*Simplification :*

* drop the reset gate
* depends only on the input xt, not on previous hidden state
* remove **tanh** on candidate state

*Performance :*

* use ~13-33% of GRU parameter
* Up to 175× faster at seq length 512, and 1324× faster at seq length 4096 without cuDNN

**minLSTM**

![minLSTM](assets/LSTMV2.png)

*Simplification :*

* Remove dependency on previous hidden state
* Drop the output gate
* Remove **tanh**
* Normalize forget/input gates so they sum to 1 (time-independent scaling).

*Performance :*

* use ~15–38% of LSTM parameter
* Up to 235× faster at seq length 512, and 1361× faster at seq length 4096

both are fully parallelized (when using parallel scan), no BPTT

## Code Details

[GRU implementation details](gru_coded_details.ipynb)

[LSTM implementation details](lstm_coded_details.ipynb)

model codes

`gru_v.py` & `lstm_v.py`

## References

Paper : [`Were RNN Is All We Needed`](https://arxiv.org/pdf/2410.01201)
