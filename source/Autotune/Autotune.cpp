// Copyright 2017 M. Bredholt and M. Kirkegaard
#include "SC_PlugIn.h"
#include <fftw3.h>
#include <complex>


#define FFT_SIZE 4096
#define FFT_COMPLEX_SIZE static_cast<int>(FFT_SIZE/2 + 1)
#define WIN_SIZE 2048
#define HOP_SIZE 1024
#define SHUNT_SIZE (FFT_SIZE - HOP_SIZE)
#define PI 3.1415926535898f
#define TWOPI 6.28318530717952646f

#define HANN_WINDOW 0
#define SINE_WINDOW 1
#define CEPSTRUM_LOWPASS 2

#define CEPSTRUM_CUTOFF static_cast<int>(0.05 * 1024)

static InterfaceTable *ft;

typedef std::complex<float> cfloat;
const cfloat im(0.0,1.0);

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

void do_log_abs(cfloat *in, int N) {
  for (int i = 0; i < N; ++i) {
    in[i] = log(std::abs(in[i]) + 1e-6);
  }
}

void do_real_exp(float *out, cfloat *in, int N) {
  for (int i = 0; i < N; ++i) {
    out[i] = exp(std::real(in[i]));
  }
}

void do_zeropad(float *in, int old_length, int new_length) {
  memset(in + old_length, 0, (new_length - old_length) * sizeof(float));
}

float do_autocorrelation(cfloat *in, float sampleRate, int N) {
  float highest_peak = -1e9;
  int index = 0;
  for (int i = 0; i < N; ++i) {
    float corr = static_cast<float>(in[i] * std::conj(in[i]));
    if (corr > highest_peak){
      highest_peak = corr;
      index = i;
    }
  }
  return index * sampleRate/N;
}

int create_pulse_train(float *in, float freq, float sampleRate, int N, int current_offset) {
  int period = static_cast<int>(sampleRate/freq + 0.5);
  int repeats = static_cast<int>((N - current_offset)/period);
  int new_offset = (repeats + 1) * period + current_offset - N;

  memset(in, 0, N * sizeof(float));
  
  in[current_offset] = 1.0;
  for (int i = current_offset + 1; i - current_offset < period; ++i) {
    in[i] = 0.0;
  }

  for (int i = 0; i < repeats; ++i) {
    memcpy(in + i * period, in, period * sizeof(float));
  }

  return new_offset;
}

struct Autotune : public Unit {
  float *win;
  float *in_buffer, *out_buffer, *fft_real, *tmp_buffer;
  cfloat *fft_complex;
  fftwf_plan fft, ifft;
  float m_fbufnum;
  SndBuf *m_buf;
  int pos;
  int oscillator_offset;
  float freq;
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

  unit->in_buffer = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));
  unit->out_buffer = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));

  unit->tmp_buffer = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));

  unit->fft_real = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));
  unit->fft_complex = static_cast<cfloat*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(cfloat)));

  memset(unit->in_buffer, 0, FFT_SIZE * sizeof(float));
  memset(unit->out_buffer, 0, FFT_SIZE * sizeof(float));
  memset(unit->fft_real, 0, FFT_SIZE * sizeof(float));
  memset(unit->fft_complex, 0, FFT_COMPLEX_SIZE * sizeof(cfloat));

  unit->fft = fftwf_plan_dft_r2c_1d(
    FFT_SIZE, unit->fft_real, reinterpret_cast<fftwf_complex*>(unit->fft_complex), FFTW_ESTIMATE);
  unit->ifft = fftwf_plan_dft_c2r_1d(
    FFT_SIZE, reinterpret_cast<fftwf_complex*>(unit->fft_complex), unit->fft_real, FFTW_ESTIMATE);

  unit->pos = 0;
  unit->oscillator_offset = 0;

  SETCALC(Autotune_next);
  Autotune_next(unit, 1);
}

void Autotune_next(Autotune *unit, int inNumSamples) {
  float *in = IN(1);
  float *out = OUT(0);
  float *in_buffer = unit->in_buffer;
  float *out_buffer = unit->out_buffer;
  float *tmp_buffer = unit->tmp_buffer;
  float *fft_real = unit->fft_real;
  float *win = unit->win;
  float freq = IN0(2);
  cfloat *fft_complex = unit->fft_complex;
  int pos = unit->pos;
  float fund_freq = 0;
  RGen &rgen = *unit->mParent->mRGen;

  GET_BUF

  memcpy(in_buffer + SHUNT_SIZE + pos, in, inNumSamples * sizeof(float));
  pos += inNumSamples;

  if (pos >= HOP_SIZE) {
    pos = 0;

    // FFT
    memcpy(fft_real, in_buffer, WIN_SIZE * sizeof(float));
    create_window(win, HANN_WINDOW, WIN_SIZE);
    do_windowing(fft_real, win, WIN_SIZE);
    fftwf_execute(unit->fft);

    // Pitch tracking
    fund_freq = do_autocorrelation(fft_complex, SAMPLERATE, FFT_COMPLEX_SIZE);
    printf("%f\n", fund_freq);


    // Move the last part to the beginning
    memmove(in_buffer, in_buffer + HOP_SIZE, SHUNT_SIZE * sizeof(float));

    // Create cepstrum
    do_log_abs(fft_complex, FFT_COMPLEX_SIZE);
    fftwf_execute(unit->ifft);

    // Window cepstrum to estimate spectral envelope
    create_window(win, CEPSTRUM_LOWPASS, FFT_SIZE);
    do_windowing(fft_real, win, FFT_SIZE, 1.0/FFT_SIZE);

    // Go back to frequency domain to prepare for filtering
    fftwf_execute(unit->fft);
    do_real_exp(tmp_buffer, fft_complex, FFT_COMPLEX_SIZE);

    for (int i = 0; i < bufFrames; ++i) {
      bufData[i] = out_buffer[i];
    }

    // Create excitation signal
    unit->oscillator_offset = create_pulse_train(fft_real, freq, SAMPLERATE, WIN_SIZE, unit->oscillator_offset);
    // Zeropad to match conditions for convolution
    do_zeropad(fft_real, WIN_SIZE, FFT_SIZE);
    fftwf_execute(unit->fft);    

    // Circular convolution
    for (int k = 0; k < FFT_COMPLEX_SIZE; ++k) {
      fft_complex[k] *= tmp_buffer[k] * std::exp(im*static_cast<cfloat>(rgen.sum3rand(2.0)));
    }

    // IFFT
    fftwf_execute(unit->ifft);
    create_window(win, HANN_WINDOW, FFT_SIZE);
    do_windowing(fft_real, win, FFT_SIZE, 1.0/FFT_SIZE);


    // !!! SILENCE !!!
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
  unit->freq = freq;
}

void Autotune_Dtor(Autotune *unit) {
  fftwf_destroy_plan(unit->fft);
  fftwf_destroy_plan(unit->ifft);

  RTFree(unit->mWorld, unit->win);
  RTFree(unit->mWorld, unit->in_buffer);
  RTFree(unit->mWorld, unit->out_buffer);
  RTFree(unit->mWorld, unit->tmp_buffer);
  RTFree(unit->mWorld, unit->fft_real);
  RTFree(unit->mWorld, unit->fft_complex);
}

PluginLoad(Autotune) {
  ft = inTable;
  DefineDtorUnit(Autotune);
}
