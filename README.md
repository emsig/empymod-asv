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
