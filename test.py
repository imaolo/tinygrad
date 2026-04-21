from tinygrad import Tensor, function
from tinygrad.helpers import Timing
Tensor.manual_seed(0)

SIZE = 4096*4096
N_LAYERS = 10

X:Tensor = Tensor.rand(N_LAYERS, 10).realize()
Y:Tensor = Tensor.rand(N_LAYERS, 10).realize()

@function(precompile=True)
def fxn(accum, x, y):
  return accum + (x*y)


def run_test(layer_bufs=False):
  n_layers = 4

  x_full = X.clone()
  y_full = Y.clone()

  if layer_bufs:
    x_buf =  x_full[0].empty_like()
    y_buf =  y_full[0].empty_like()

  accum: Tensor = X[0].zeros_like()
  for i in range(n_layers):
    x_loc, y_loc = x_full[i], y_full[i]
    if layer_bufs:
      x_loc = x_buf.assign(x_loc)
      y_loc = y_buf.assign(y_loc)
    accum = fxn(accum, x_loc, y_loc)
    if layer_bufs:
      accum.realize()
  
  print(accum.mean().item())
  

run_test()
run_test()
with Timing(): run_test()
run_test(layer_bufs=True)
run_test(layer_bufs=True)
with Timing(): run_test(layer_bufs=True)