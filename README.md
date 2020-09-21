rfopt
=====

This crate implements a black box optimization algorithm using random forest as the surrogate model.

`rfopt` can be used as a [kurobako](https://github.com/sile/kurobako) solver as follows:
```console
$ cargo install --path .
$ kurobako studies --solvers $(kurobako solver command rfopt) --problems $(kurobako problem-suite sigopt auc) --repeats 30 --budget 100 | kurobako run > result.json
```

Benchmark Results
-----------------

### sigopt/evalset

#### Benchmark command

```console
$ kurobako studies --solvers $(kurobako solver random) $(kurobako solver command rfopt) $(kurobako solver --name TPE optuna) --problems $(kurobako problem-suite sigopt auc) --repeats 30 --budget 100 | kurobako run -p 8 > result.json
```

#### Result: summary

| Solver | Borda | Firsts |
|:-------|------:|-------:|
| Random |     0 |      3 |
| TPE    |    43 |     19 |
| rfopt  |    60 |     36 |

#### Result: detail

See [the gist](https://gist.github.com/sile/005fa9302b8f1ee5c0baf800b4538b91).

### HPOBench

#### Benchmark command

```console
$ kurobako studies --solvers $(kurobako solver random) $(kurobako solver command rfopt) $(kurobako solver --name TPE optuna --pruner NopPruner) --problems $(kurobako problem-suite hpobench fcnet .) --repeats 30 --budget 100 | kurobako run -p 8 > result.json
```

#### Result: summary

| Solver | Borda | Firsts |
|:-------|------:|-------:|
| Random |     0 |      0 |
| TPE    |     4 |      0 |
| rfopt  |     8 |      4 |

#### Result: detail

See [the gist](https://gist.github.com/sile/ffac1229e79da8d7c330778a8ce6dc0e).
