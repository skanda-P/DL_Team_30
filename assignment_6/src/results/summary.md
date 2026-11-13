# Assignment 6 Results Summary

## Dataset: `handwritten`

- **Stage 1 Paired Comparison (LSTM - RNN across 9 architectures):** Val delta = `+10.79%`, Test delta = `+6.78%`
- **Best `RNN`** (`stage2_opt`): `hidden=64, layers=2, bi=False, do=0.2, opt=adam, lr=0.001` | params=`12997`
  - Train=`100.00%`, Val=`91.30%`, Test=`96.00%`, Epochs=`1770` (converged)
  - `RNN` sweep: `15` configs, `0` non-converged, `0` diverged.
- **Best `LSTM`** (`stage2_opt`): `hidden=128, layers=3, bi=False, do=0.0, opt=rmsprop, lr=0.001` | params=`332421`
  - Train=`100.00%`, Val=`98.55%`, Test=`100.00%`, Epochs=`97` (converged)
  - `LSTM` sweep: `15` configs, `0` non-converged, `0` diverged.

## Dataset: `cv`

- **Stage 1 Paired Comparison (LSTM - RNN across 9 architectures):** Val delta = `+3.03%`, Test delta = `+3.47%`
- **Best `RNN`** (`stage2_opt`): `hidden=128, layers=3, bi=False, do=0.0, opt=sgd, lr=0.01` | params=`88325`
  - Train=`100.00%`, Val=`94.92%`, Test=`92.02%`, Epochs=`69` (converged)
  - `RNN` sweep: `15` configs, `0` non-converged, `0` diverged.
- **Best `LSTM`** (`stage2_opt`): `hidden=128, layers=3, bi=True, do=0.0, opt=adam, lr=0.001` | params=`964869`
  - Train=`98.93%`, Val=`95.73%`, Test=`95.60%`, Epochs=`10` (converged)
  - `LSTM` sweep: `15` configs, `0` non-converged, `0` diverged.

