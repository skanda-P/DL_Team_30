# Assignment 6 Results Summary

## Dataset: `handwritten`

- **Stage 1 Paired Comparison (LSTM - RNN across 9 architectures):** Val delta = `+22.54%`, Test delta = `+17.33%`
- **Best `RNN`** (`stage2_opt`): `hidden=64, layers=1, bi=True, do=0.0, opt=adam, lr=0.001` | params=`9349`
  - Train=`100.00%`, Val=`94.20%`, Test=`99.00%`, Epochs=`364` (converged)
  - `RNN` sweep: `15` configs, `3` non-converged, `0` diverged.
- **Best `LSTM`** (`stage1_arch`): `hidden=32, layers=1, bi=False, do=0.0, opt=adam, lr=0.001` | params=`4773`
  - Train=`98.91%`, Val=`98.55%`, Test=`97.00%`, Epochs=`484` (converged)
  - `LSTM` sweep: `15` configs, `0` non-converged, `0` diverged.

## Dataset: `cv`

- **Stage 1 Paired Comparison (LSTM - RNN across 9 architectures):** Val delta = `+3.16%`, Test delta = `+3.76%`
- **Best `RNN`** (`stage1_arch`): `hidden=128, layers=2, bi=False, do=0.0, opt=adam, lr=0.001` | params=`55301`
  - Train=`100.00%`, Val=`94.72%`, Test=`91.86%`, Epochs=`64` (converged)
  - `RNN` sweep: `15` configs, `0` non-converged, `0` diverged.
- **Best `LSTM`** (`stage2_opt`): `hidden=128, layers=2, bi=False, do=0.0, opt=rmsprop, lr=0.001` | params=`219269`
  - Train=`100.00%`, Val=`96.34%`, Test=`95.77%`, Epochs=`48` (converged)
  - `LSTM` sweep: `15` configs, `0` non-converged, `0` diverged.

