rfopt
=====

This crate implements a black box optimization algorithm using random forest as the surrogate model.

`rfopt` can be used as a [kurobako](https://github.com/sile/kurobako) solver as follows:
```console
$ cargo install --path .
$ kurobako studies --solvers $(kurobako solver command rfopt) --problems (kurobako problem-suite sigopt auc) --repeats 30 --budget 100 | kurobako run > result.json
```
