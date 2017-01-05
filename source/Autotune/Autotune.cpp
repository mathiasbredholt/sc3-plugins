// Copyright 2017 M. Bredholt and M. Kirkegaard
#include "SC_PlugIn.h"
#include <fftw3.h>

#define FFT_SIZE 64

static InterfaceTable *ft;

struct Autotune : public Unit {
  double *fft_in;
  fftw_complex *fft_out;
  fftw_plan p;
};

extern "C" {
  void Autotune_next(Autotune *unit, int inNumSamples);
  void Autotune_Ctor(Autotune *unit);
  void Autotune_Dtor(Autotune *unit);
}

void Autotune_Ctor(Autotune *unit) {
  unit->fft_in = (double*) fftw_malloc(sizeof(double) * FFT_SIZE);
  unit->fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * FFT_SIZE);
  unit->p = fftw_plan_dft_r2c_1d(
    FFT_SIZE, unit->fft_in, unit->fft_out, FFTW_ESTIMATE);

  // unit->buf = (float*) RTAlloc(unit->mWorld, 20 * sizeof(float));
  // memset(unit->buf, 0, 20 * sizeof(float));

  SETCALC(Autotune_next);
  Autotune_next(unit, 1);
}

void Autotune_next(Autotune *unit, int inNumSamples) {
  float *in = IN(0);
  float *out = OUT(0);
  // float *buf = unit->buf;


  // 10th order moving average
  // float tmp;

  // for (int i = 0; i < inNumSamples; ++i)
  // {
  //   tmp = 0;

  //   for (int j = 0; j < 10; j++)
  //   {
  //     if ((i - j) >= 0)
  //     {
  //       // unit->buf[j] = in[i - j];
  //       tmp += 1/10.0 * in[i - j];
  //     }
  //   }

  //   out[i] = tmp;

  // }

  unit->fft_in = (double*) in;
  fftw_execute(unit->p);
  out = (float*) unit->fft_out;
}

void Autotune_Dtor(Autotune *unit) {
  // RTFree(unit->mWorld, unit->buf);
  fftw_destroy_plan(unit->p);
  fftw_free(unit->fft_in);
  fftw_free(unit->fft_out);
}

PluginLoad(Autotune) {
  ft = inTable;
  DefineDtorUnit(Autotune);
}
