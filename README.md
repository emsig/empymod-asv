# Benchmarks for empymod using airspeed velocity

[![asv](http://img.shields.io/badge/benchmarked%20by-asv-brightgreen.svg?style=flat)](http://empymod.github.io/asv/)

Currently there are benchmarks for:

   - `kernel`: `wavenumber`, `greenfct`, `reflections`, `fields`;
   - `transform`: `fht`, `hqwe`, `hquad`, `ffht`, `fqwe`, `fftlog`, `fft`,
                  `dlf`;
   - `model`: `bipole`, `dipole`, `analytical`.

The benchmarks are not very extensive, but sufficiently enough to catch any
serious regression.

Most benchmarks only work backwards until v1.2.0. Before, `numexpr` was a
mandatory requirement.
