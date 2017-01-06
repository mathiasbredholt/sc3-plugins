// Copyright 2017 M. Bredholt and M. Kirkegaard
#include "SC_PlugIn.h"
#include <fftw3.h>
#include <complex>


#define FFT_SIZE 2048
#define HOP_SIZE 512
#define SHUNT_SIZE (FFT_SIZE - HOP_SIZE)
#define PI 3.1415926535898f
#define TWOPI 6.28318530717952646f

#define HANN 0
#define SINE 1

static InterfaceTable *ft;

void create_window(float *win, int wintype, int N) {
  if (wintype == HANN) {
    for (int i = 0; i < N; ++i) {
      win[i] = 0.5 - 0.5 * cos(TWOPI*i/(N-1));
    }
  } else if (wintype == SINE) {
    for (int i = 0; i < N; ++i) {
      win[i] = sin(TWOPI*i/N);
    }
  }
}

void do_windowing(float *in, float *win, int N, float scale = 1.0) {
  for (int i = 0; i < N; ++i) {
    in[i] *= win[i] * scale;
  }
}

float* log_abs(std::complex<float> *in, int N) {
  for (int i = 0; i < N; ++i) {
    in[i] = log(std::abs(in[i]));
  }
}

struct Autotune : public Unit {
  float *win;
  float *in_buffer, *out_buffer, *fft_real;
  std::complex<float> *fft_complex;
  fftwf_plan fft, ifft;
  int writepos;
  int readpos;
};

extern "C" {
  void Autotune_next(Autotune *unit, int inNumSamples);
  void Autotune_Ctor(Autotune *unit);
  void Autotune_Dtor(Autotune *unit);
}

void Autotune_Ctor(Autotune *unit) {
  // Create window function
  unit->win = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));

  create_window(unit->win, SINE, FFT_SIZE);

  unit->in_buffer = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));
  unit->out_buffer = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));


  unit->fft_real = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));
  unit->fft_complex = static_cast<std::complex<float>*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(std::complex<float>)));

  memset(unit->in_buffer, 0, FFT_SIZE * sizeof(float));
  memset(unit->out_buffer, 0, FFT_SIZE * sizeof(float));
  memset(unit->fft_real, 0, FFT_SIZE * sizeof(float));
  memset(unit->fft_complex, 0, FFT_SIZE * sizeof(std::complex<float>));

  unit->fft = fftwf_plan_dft_r2c_1d(
    FFT_SIZE, unit->fft_real, reinterpret_cast<fftwf_complex*>(unit->fft_complex), FFTW_ESTIMATE);
  unit->ifft = fftwf_plan_dft_c2r_1d(
    FFT_SIZE, reinterpret_cast<fftwf_complex*>(unit->fft_complex), unit->fft_real, FFTW_ESTIMATE);

  unit->writepos = 0;
  unit->readpos = 0;

  SETCALC(Autotune_next);
  Autotune_next(unit, 1);
}

void Autotune_next(Autotune *unit, int inNumSamples) {
  float *in = IN(0);
  float *out = OUT(0);
  float *in_buffer = unit->in_buffer;
  float *out_buffer = unit->out_buffer;
  float *fft_real = unit->fft_real;
  float *win = unit->win;
  std::complex<float> *fft_complex = unit->fft_complex;
  std::complex<float> *tmp = static_cast<std::complex<float>*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(fftwf_complex)));
  int writepos = unit->writepos;
  int readpos = unit->readpos;

  memcpy(in_buffer + SHUNT_SIZE + writepos, in, inNumSamples * sizeof(float));
  writepos += inNumSamples;

  if (writepos >= HOP_SIZE) {
    writepos = 0;

    // // FFT
    memcpy(fft_real, in_buffer, FFT_SIZE * sizeof(float));
    do_windowing(fft_real, win, FFT_SIZE);
    fftwf_execute(unit->fft);

    // Move the last part to the beginning
    memmove(in_buffer, in_buffer + HOP_SIZE, SHUNT_SIZE * sizeof(float));

    memcpy(tmp, fft_complex, FFT_SIZE * sizeof(std::complex<float>));


    log_abs(fft_complex, unit->ifft, FFT_SIZE);
    fftwf_execute(unit->ifft);
    // fft_complex is now a cepstrum




    // // IFFT
    memcpy(fft_complex, tmp, FFT_SIZE * sizeof(fftwf_complex));
    fftwf_execute(unit->ifft);
    do_windowing(fft_real, win, FFT_SIZE, 1.0/FFT_SIZE);

    memmove(out_buffer, out_buffer + HOP_SIZE, SHUNT_SIZE * sizeof(float));

    for (int i = 0; i < FFT_SIZE; ++i) {
      if (i < SHUNT_SIZE) {
        // Overlap and add
        out_buffer[i] += fft_real[i];
      } else {
        // Copy
        out_buffer[i] = fft_real[i];
      }
    }

    readpos = 0;
  }

  memcpy(out, out_buffer + readpos, inNumSamples * sizeof(float));
  readpos += inNumSamples;


  // if (readpos >= FFT_SIZE) {
  //   readpos = 0;
  // }

  unit->writepos = writepos;
  unit->readpos = readpos;
}

void Autotune_Dtor(Autotune *unit) {
  fftwf_destroy_plan(unit->fft);
  fftwf_destroy_plan(unit->ifft);

  RTFree(unit->mWorld, unit->win);
  RTFree(unit->mWorld, unit->in_buffer);
  RTFree(unit->mWorld, unit->out_buffer);
  RTFree(unit->mWorld, unit->fft_real);
  RTFree(unit->mWorld, unit->fft_complex);
}

PluginLoad(Autotune) {
  ft = inTable;
  DefineDtorUnit(Autotune);
}
