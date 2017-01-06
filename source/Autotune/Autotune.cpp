// Copyright 2017 M. Bredholt and M. Kirkegaard
#include "SC_PlugIn.h"
#include <fftw3.h>
#include <complex>


#define FFT_SIZE 2048
#define WIN_SIZE 1024
#define HOP_SIZE 512
#define SHUNT_SIZE (FFT_SIZE - HOP_SIZE)
#define PI 3.1415926535898f
#define TWOPI 6.28318530717952646f

#define HANN_WINDOW 0
#define SINE_WINDOW 1
#define CEPSTRUM_LOWPASS 2

#define CEPSTRUM_CUTOFF 100

static InterfaceTable *ft;

void create_window(float *win, int wintype, int N) {
  if (wintype == HANN_WINDOW) {
    for (int i = 0; i < N; ++i) {
      win[i] = 0.5 - 0.5 * cos(TWOPI*i/(N-1));
    }
  } else if (wintype == SINE_WINDOW) {
    for (int i = 0; i < N; ++i) {
      win[i] = sin(TWOPI*i/N);
    }
  } else if (wintype == CEPSTRUM_LOWPASS) {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == CEPSTRUM_CUTOFF) {
        win[i] = 1.0;
      } else if (1 <= i && i < CEPSTRUM_CUTOFF) {
        win[i] = 2.0;
      } else if (CEPSTRUM_CUTOFF < i && i <= N - 1) {
        win[i] = 0.0;
      }
    }
  }
}

void do_windowing(float *in, float *win, int N, float scale = 1.0) {
  for (int i = 0; i < N; ++i) {
    in[i] *= win[i] * scale;
  }
}

void do_log_abs(std::complex<float> *in, int N) {
  for (int i = 0; i < N; ++i) {
    in[i] = log(std::abs(in[i]) + 1e-6);
  }
}

void do_real_exp(float *out, std::complex<float> *in, int N) {
  for (int i = 0; i < N; ++i) {
    out[i] = exp(std::real(in[i]));
  }
}

void do_zeropad(float *in, int old_length, int new_length) {
  memset(in + old_length, 0, (new_length - old_length) * sizeof(float));
}

void create_pulse_train(float *in, float freq, float sampleRate, int N) {
  int period = static_cast<int>(sampleRate/freq);
  int repeats = static_cast<int>(N/period);
  memset(in, 0, N * sizeof(float));
  in[0] = 1.0;
  for (int i = 1; i < period; ++i) {
    in[i] = 0.0;
  }

  for (int i = 0; i < repeats; ++i) {
    memcpy(in + i * period, in, period * sizeof(float));
  }
}

struct Autotune : public Unit {
  float *win;
  float *in_buffer, *out_buffer, *fft_real;
  std::complex<float> *fft_complex;
  fftwf_plan fft, ifft;
  float m_fbufnum;
  SndBuf *m_buf;
  int pos;
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

  unit->m_fbufnum = -1e-9;

  create_window(unit->win, SINE_WINDOW, FFT_SIZE);

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

  unit->pos = 0;

  SETCALC(Autotune_next);
  Autotune_next(unit, 1);
}

void Autotune_next(Autotune *unit, int inNumSamples) {
  float *in = IN(1);
  float *out = OUT(0);
  float *in_buffer = unit->in_buffer;
  float *out_buffer = unit->out_buffer;
  float *fft_real = unit->fft_real;
  float *win = unit->win;
  std::complex<float> *fft_complex = unit->fft_complex;
  int pos = unit->pos;

  GET_BUF

  memcpy(in_buffer + SHUNT_SIZE + pos, in, inNumSamples * sizeof(float));
  pos += inNumSamples;

  if (pos >= HOP_SIZE) {
    pos = 0;

    // FFT
    memcpy(fft_real, in_buffer, WIN_SIZE * sizeof(float));
    do_windowing(fft_real, win, WIN_SIZE);
    fftwf_execute(unit->fft);

    // Move the last part to the beginning
    memmove(in_buffer, in_buffer + HOP_SIZE, SHUNT_SIZE * sizeof(float));

    // Create cepstrum
    do_log_abs(fft_complex, FFT_SIZE);
    fftwf_execute(unit->ifft);

    // Window cepstrum to estimate spectral envelope
    create_window(win, CEPSTRUM_LOWPASS, FFT_SIZE);
    do_windowing(fft_real, win, FFT_SIZE, 1.0/FFT_SIZE);

    // Go back to frequency domain to prepare for filtering
    fftwf_execute(unit->fft);
    do_real_exp(out_buffer, fft_complex, FFT_SIZE);

    // Create excitation signal
    create_pulse_train(fft_real, 200, SAMPLERATE, WIN_SIZE);
    // Zeropad to match conditions for convolution
    do_zeropad(fft_real, WIN_SIZE, FFT_SIZE);
    fftwf_execute(unit->fft);

    // Circular convolution
    for (int i = 0; i < FFT_SIZE; ++i) {
      fft_complex[i] *= out_buffer[i];
    }

    // IFFT
    fftwf_execute(unit->ifft);
    create_window(win, SINE_WINDOW, FFT_SIZE);
    do_windowing(fft_real, win, FFT_SIZE, 0.5/FFT_SIZE);

    for (int i = 0; i < bufFrames; ++i) {
      bufData[i] = fft_real[i];
    }

    // memset(fft_real, 0, FFT_SIZE * sizeof(float));
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

  }

  memcpy(out, out_buffer + pos, inNumSamples * sizeof(float));
  // readpos += inNumSamples;


  // if (readpos >= FFT_SIZE) {
  //   readpos = 0;
  // }

  unit->pos = pos;
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
