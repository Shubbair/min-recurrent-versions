import torch

import time

def gru_tests(model1, model2):
    B, T, input_dim, hidden_dim = 64, 200, 128, 256
    print(f'B={B} , T={T} , input dimension={input_dim}, hidden dimemnsion={hidden_dim}')

    # Random input and initial hidden state
    x = torch.randn(B, T, input_dim)
    h0 = torch.zeros(B, hidden_dim)

    # Time the standard GRU
    gru = model1(input_dim, hidden_dim, batch_first=True)
    start_time = time.time()
    out_gru, h_n_gru = gru(x)
    end_time = time.time()

    print(f"[{model1.__name__}] Time taken: {end_time - start_time:.6f} seconds")


    # Time the custom cell
    cell = model2(input_dim, hidden_dim)
    h = h0
    outputs = []
    start_time = time.time()
    for t in range(T):
        h = cell(x[:, t, :], h)
        outputs.append(h.unsqueeze(1))
    out_custom = torch.cat(outputs, dim=1)
    end_time = time.time()
    print(f"[recoded {model2.__name__}] Time taken: {end_time - start_time:.6f} seconds")

    # Check if shapes match
    assert out_gru.shape == out_custom.shape, "Output shapes do not match!"
    assert h_n_gru.shape == h.unsqueeze(0).shape, "Final hidden shapes do not match!"
    
    print('input : ', x.shape)
    print('output : ', out_gru.shape)

    print(f"Shapes matched.")
    
def lstm_tests(model1, model2):
    B, T, input_dim, hidden_dim = 64, 200, 128, 256
    print(f'B={B} , T={T} , input dimension={input_dim}, hidden dimemnsion={hidden_dim}')

    # Random input and initial hidden and cell states
    x = torch.randn(B, T, input_dim)
    h0 = torch.zeros(B, hidden_dim)
    c0 = torch.zeros(B, hidden_dim)

    # Time the standard LSTM
    lstm = model1(input_dim, hidden_dim, batch_first=True)
    start_time = time.time()
    out_lstm, (h_n_lstm, c_n_lstm) = lstm(x)
    end_time = time.time()

    print(f"[{model1.__name__}] Time taken: {end_time - start_time:.6f} seconds")

    # Time the custom cell
    cell = model2(input_dim, hidden_dim)
    h, c = h0, c0
    outputs = []
    start_time = time.time()
    for t in range(T):
        h, c = cell(x[:, t, :], h, c)
        outputs.append(h.unsqueeze(1))
    out_custom = torch.cat(outputs, dim=1)
    end_time = time.time()
    print(f"[recoded {model2.__name__}] Time taken: {end_time - start_time:.6f} seconds")

    # Check if shapes match
    assert out_lstm.shape == out_custom.shape, "Output shapes do not match!"
    assert h_n_lstm.shape == h.unsqueeze(0).shape, "Final hidden shapes do not match!"
    assert c_n_lstm.shape == c.unsqueeze(0).shape, "Final cell shapes do not match!"

    print('input : ', x.shape)
    print('output : ', out_lstm.shape)
    print(f"Shapes matched.")
