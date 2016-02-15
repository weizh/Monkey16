---
-- LSTM cell.
-- Supports vector input (in mini-batches also) and one hot encoding index (in mini-batches also)
--
-- input_size : input vector length (onehot mode does not have this information, so this is needed.)
--              This is not needed though if vector as input.
--
-- rnn_size   : size of the internal activations
--
-- n          : number of layers.
--
-- dropout    : dropout rate. Dropout is applied both on input and output.
--
-- onehot_flag     : a flag, True for using one hot. False for using vector.
--
-- output_flag: a flag, True if using output_dist logSoftMax.
-- 
-- batch_norm_flag: a flag, True if using batch normalization over WX.
--
-- Inputs: 2n+1 inputs: {x, p_c_1 , p_h_1, p_c_2, p_h_2,... p_c_n, p_h_n}
--
-- Outputs: if output_flag:  2n+1 outputs, { c_1, h_1, c_2, h_2, ..., c_n, h_n, out } where out = logSoftMax( W * dropout (h_n) )
--          else: 2n outputs, { c_1, h_1, c_2, h_2, ..., c_n, h_n } where no out exists.
---

local LSTM = {}
function LSTM.lstm(input_size, output_size, rnn_size, n, dropout, onehot_flag, output_flag)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      if onehot_flag then
        x = OneHot(input_size)(inputs[1])
      else
        x = inputs[1]
      end
      input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  if output_flag then
    local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
  end
  return nn.gModule(inputs, outputs)
end

return LSTM

