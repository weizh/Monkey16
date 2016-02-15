---
--AP-CNN described in Cicero et al. 2015. Attentive Pooling Networks.
--
--cnn_size: The size of the cnn matrix U. Or, the word vector size in QA problem
--

local AP_CNN = {}
function AP_CNN.apcnn(cnn_size)
  
  local D = nn.Identity()() -- batch size * doclen * wlen
  local Q = nn.Identity()()  -- batch_size * qlen * wlen

  local inputs = {D,Q}
    
  local UQ = nn.Linear(cnn_size,cnn_size)(Q)
  local DUQ = nn.MM(true,false)({D,UQ})
    
  -- Take max ALONG the given dimension. Assume 2d input. batch is first dimension.
  local dpool = nn.Max(2,2)(DUQ)
  local qpool = nn.Max(1,2)(DUQ)
  
  local dsoftmax = nn.SoftMax(dpool)
  local qsoftmax = nn.SoftMax(qpool)
  
  local rd = nn.MM()({D,dsoftmax})
  local rq = nn.MM()({Q,qsoftmax})
  
  local outputs = {dsoftmax,rd, qsoftmax, rq}
  return nn.gModule({inputs,outputs})  
end

return AP_CNN
