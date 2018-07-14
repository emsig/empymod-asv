# Benchmarks for empymod using airspeed velocity

[![asv](http://img.shields.io/badge/benchmarked%20by-asv-brightgreen.svg?style=flat)](http://empymod.github.io/asv/)

Currently there are benchmarks for:

   - `kernel`: `wavenumber`, `greenfct`, `reflections`, `fields`;
   - `transform`: `fht`, `hqwe`, `hquad`, `ffht`, `fqwe`, `fftlog`, `fft`,
     `dlf`;
   - `model`: `bipole`, `dipole`, `analytical`.

The benchmarks are not very extensive, but sufficiently enough to catch any
serious regression. The results have to be interpreted carefully, taking the
whole modelling chain into account (calculation as well as Hankel- and Fourier
transforms and interpolation, if applicable). An improvement or a regression in
one of the functions does not necessarily mean an improvement or a regression
in the overall calculation (specifically true for the functions in
`transform`).

The benchmarks only work backwards until a bit before `v1.2.0` (`a1dbe8ef` to
be precise). Before, `numexpr` was a mandatory requirement but it was not in
the `requirements.txt` and is not installed. If you want to run the benchmarks
for earlier commits too, you have to install `numexpr` manually in the
environment which is used by `asv` (in the `asv/env`-directory). No benchmarks
are carried out for `numexpr`, as `numexpr` really only helps if `empymod` is
run in parallel; benchmarks are run single-threaded, so there is not much
difference between `opt=None` and `opt='parallel'`.

In theory `asv` does not know about threading, everything is run with a single
thread. However, there are cases where `OpenMP/OpenBLAS/MKL` might try some
"smart" threading. On my machine, three benchmarks immediately send the CPU
usage to 100%. However, instead of speeding things up this attempt of using
several threads slows things down. Ensure `asv` is only using one thread on
your machine. In my case, the following resolved the issue (see `asv`-issue
[#671](https://github.com/airspeed-velocity/asv/issues/671).)

```
export OMP_NUM_THREADS=1
```

The results which are shown on
[empymod.github.io/asv](http://empymod.github.io/asv/) are stored in the
[github.com/empymod/bench](http://github.com/empymod/bench)-repo.
