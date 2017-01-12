// Copyright 2017 M. Bredholt and M. Kirkegaard
#include "SC_PlugIn.h"
#include <fftw3.h>
#include <complex>

#define FFT_SIZE 4096
#define FFT_COMPLEX_SIZE static_cast<int>(FFT_SIZE/2 + 1)
#define WIN_SIZE 2048
#define HOP_SIZE 512
#define SHUNT_SIZE (FFT_SIZE - HOP_SIZE)
#define PI 3.1415926535898f
#define TWOPI 6.28318530717952646f
#define EXPECTED_PHASE_DIFF (TWOPI * HOP_SIZE/WIN_SIZE)
#define CORRELATION_CONSTANT 0.8

#define HANN_WINDOW 0
#define SINE_WINDOW 1
#define CEPSTRUM_LOWPASS 2

#define CEPSTRUM_CUTOFF static_cast<int>(0.1 * WIN_SIZE)

static InterfaceTable *ft;

typedef std::complex<float> cfloat;


const cfloat im(0.0, 1.0);

const float freq_grid[] = {
  130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
  261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88
};

void create_window(float *win, int wintype, int N) {
  if (wintype == HANN_WINDOW) {
    // Hann window squared
    for (int i = 0; i <   N; ++i) {
      win[i] = pow((0.5 - 0.5 * cos(TWOPI*i/(N-1))), 2);
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

float do_autocorrelation(cfloat *in, float *phase_buffer,
                         float sample_rate, int N) {
  float highest_peak = -1e9;
  int index = 0;
  float trueFreq = 0;
  float diff;
  int tmp;
  float freq_per_bin = sample_rate / (2.0 * N);

  for (int i = 0; i < N; ++i) {
    // Phase calculations
    diff = std::arg(in[i]) - phase_buffer[i];
    phase_buffer[i] = std::arg(in[i]);
    diff -= static_cast<float>(i) * EXPECTED_PHASE_DIFF;

    tmp = diff / PI;
    if (tmp >= 0) {
      tmp += tmp & 1;
    } else {
      tmp -= tmp & 1;
    }
    diff -= PI * tmp;

    /* get deviation from bin frequency from the +/- Pi interval */
    diff = static_cast<float>(HOP_SIZE) / WIN_SIZE * diff / TWOPI;

    /* compute the k-th partials' true frequency */
    diff = static_cast<float>(i) * freq_per_bin + diff * freq_per_bin;

    float corr = std::real(in[i] * std::conj(in[i]));
    if (corr > highest_peak) {
      highest_peak = corr;
      index = i;
      trueFreq = diff;
    }
  }
  return trueFreq;
}

void calc_power_spectral_density(cfloat *spectrum, int N) {
  for (int i = 0; i < N; ++i) {
    spectrum[i] *= std::conj(spectrum[i]);
  }
}

void calc_square_difference_function(float *autocorrelation, 
                                     float *in_buffer, int N) {
  float sum_squared = autocorrelation[0];
  float sum_sq_left = sum_squared, sum_sq_right = sum_squared;

  for (int i = 0; i < (N+1)/2; ++i) {
    sum_sq_left  -= pow(in_buffer[i], 2);
    sum_sq_right -= pow(in_buffer[N-1-i], 2);
    autocorrelation[i] *= 2.0 / (sum_sq_left + sum_sq_right);
  }

}

void do_peak_picking(float *sdf, float sample_rate, int N, float *freq, float *corr) {
  int i = 0;
  int n = 0;
  int peaks[64];
  int local_max;
  int global_max;

  while (n < 64 && i < (N + 1) / 2 - 1) {
    // Find positively sloped zero-crossing
    if (sdf[i] < 0 && sdf[i + 1] >= 0) {
      local_max = i;
      // Find largest local maxima
      while (n < 64 && i < (N + 1) / 2 - 1) {
        // Current value is a local maxima within zero-crossings
        if (sdf[i] > sdf[local_max] && sdf[i] > sdf[i-1] && sdf[i] > sdf[i+1]) {
          local_max = i;
        }
        // Negatively sloped zero-crossing detected - add peak to list
        if (sdf[i] > 0 && sdf[i + 1] <= 0) {
          peaks[n] = local_max;
          n++;
          break;
        }
        i++;
      }
      // No zero-crossing was found - save last peak
      if (i == (N + 1) / 2 - 1) {
        peaks[n] = local_max;
        n++;
      }
    }
    i++;
  }

  if (n == 0) {
    *corr = 0;
    return;
  }


  // Find global max
  global_max = peaks[0];

  for (int j = 1; j < n - 1; ++j) {
    if (sdf[peaks[j]] > sdf[global_max]) {
      global_max = peaks[j];
    }
  }

  // Find first peak above thresh
  for (i = 0; i < n - 1; ++i) {
    if (sdf[peaks[i]] > sdf[global_max] * CORRELATION_CONSTANT) {
      break;
    }
  }

  *corr = sdf[peaks[i]];
  if (*corr > 0.5)
    *freq = sample_rate / static_cast<float>(peaks[i]);
}

float closest_frequency(float freq) {
  int k = 0;
  float diff = 1e6;
  for (int i = 0; i < 5; ++i) {
    if (abs(freq - freq_grid[i]) < diff) {
      diff = abs(freq - freq_grid[i]);
      k = i;
    }
  }
  return freq_grid[k];
}

int create_pulse_train(float *in, float freq, float sample_rate,
                       int N, int current_offset) {
  int period = static_cast<int>(sample_rate/freq + 0.5);
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

void do_pitch_shift(cfloat *new_spectrum, cfloat *orig_spectrum,
                    float *spectral_env, float ratio, int N) {
  int new_index;
  memset(new_spectrum, 0, N * sizeof(cfloat));
  for (int i = 0; i < N; ++i) {
    new_index = static_cast<int>(i * ratio);
    if (new_index >= N) break;
    // new_spectrum[new_index] += abs(orig_spectrum[i]);
    new_spectrum[new_index] += abs(orig_spectrum[i]) *
    // std::exp(im * std::arg(orig_spectrum[i])) *
    static_cast<cfloat>(spectral_env[new_index]/spectral_env[i]);
  }
}

void psola_analysis() {

}

void psola_pitch_shift() {

}

struct Autotune : public Unit {
  float *win;
  float *in_buffer, *out_buffer, *fft_real, *tmp_buffer, *phase_buffer;
  cfloat *fft_complex, *spectrum;
  fftwf_plan fft, ifft;
  float m_fbufnum;
  SndBuf *m_buf;
  int pos;
  int oscillator_offset;
  float freq;
  float fund_freq;
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
  unit->phase_buffer = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));

  unit->fft_real = static_cast<float*>(
    RTAlloc(unit->mWorld, FFT_SIZE * sizeof(float)));
  unit->fft_complex = static_cast<cfloat*>(
    RTAlloc(unit->mWorld, FFT_COMPLEX_SIZE * sizeof(cfloat)));

  unit->spectrum = static_cast<cfloat*>(
    RTAlloc(unit->mWorld, FFT_COMPLEX_SIZE * sizeof(cfloat)));

  memset(unit->in_buffer, 0, FFT_SIZE * sizeof(float));
  memset(unit->out_buffer, 0, FFT_SIZE * sizeof(float));
  memset(unit->fft_real, 0, FFT_SIZE * sizeof(float));
  memset(unit->fft_complex, 0, FFT_COMPLEX_SIZE * sizeof(cfloat));

  unit->fft = fftwf_plan_dft_r2c_1d(
    FFT_SIZE,
    unit->fft_real,

    reinterpret_cast<fftwf_complex*>(unit->fft_complex),
    FFTW_ESTIMATE);
  unit->ifft = fftwf_plan_dft_c2r_1d(
    FFT_SIZE,
    reinterpret_cast<fftwf_complex*>(unit->fft_complex),
    unit->fft_real,
    FFTW_ESTIMATE);

  unit->pos = 0;
  unit->oscillator_offset = 0;
  unit->fund_freq = 440.0;

  SETCALC(Autotune_next);
  Autotune_next(unit, 1);
}

void Autotune_next(Autotune *unit, int inNumSamples) {
  float *in = IN(1);
  float *out = OUT(0);
  float *in_buffer = unit->in_buffer;
  float *out_buffer = unit->out_buffer;
  float *tmp_buffer = unit->tmp_buffer;
  float *phase_buffer = unit->phase_buffer;
  float *fft_real = unit->fft_real;
  float *win = unit->win;
  float freq = IN0(2);
  cfloat *fft_complex = unit->fft_complex;
  cfloat *spectrum = unit->spectrum;
  int pos = unit->pos;
  float fund_freq = unit->fund_freq;
  float new_freq = 1;
  float corr = 0;

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

    // Save spectrum for later
    memcpy(spectrum, fft_complex, FFT_COMPLEX_SIZE * sizeof(cfloat));

    // ++++ Pitch tracking ++++
    // fund_freq = do_autocorrelation(fft_complex, phase_buffer,
                                   // sample_rate, FFT_COMPLEX_SIZE);

    calc_power_spectral_density(fft_complex, FFT_COMPLEX_SIZE);
    fftwf_execute(unit->ifft);
    calc_square_difference_function(fft_real, in_buffer, WIN_SIZE);
    do_peak_picking(fft_real, SAMPLERATE, WIN_SIZE, &fund_freq, &corr);

    new_freq = closest_frequency(fund_freq);
    // printf("%f\n", new_freq/fund_freq);

    // +++++++++++++++++++++++

    // ++++ Spectral envelope estimation ++++

    // Create cepstrum
    memcpy(fft_complex, spectrum, FFT_COMPLEX_SIZE * sizeof(cfloat));
    do_log_abs(fft_complex, FFT_COMPLEX_SIZE);
    fftwf_execute(unit->ifft);

    // Window cepstrum to estimate spectral envelope
    create_window(win, CEPSTRUM_LOWPASS, FFT_SIZE);
    do_windowing(fft_real, win, FFT_SIZE, 1.0/FFT_SIZE);

    // Go back to frequency domain to prepare for filtering
    fftwf_execute(unit->fft);
    do_real_exp(tmp_buffer, fft_complex, FFT_COMPLEX_SIZE);
    // // +++++++++++++++++++++++++++++++++++++++


    // // Create excitation signal
    // unit->oscillator_offset = create_pulse_train(fft_real, new_freq, SAMPLERATE,
    // WIN_SIZE, unit->oscillator_offset);
    // // Zeropad to match conditions for convolution
    // do_zeropad(fft_real, WIN_SIZE, FFT_SIZE);
    // fftwf_execute(unit->fft);
    // // Circular convolution
    // for (int k = 0; k < FFT_COMPLEX_SIZE; ++k) {
    //   fft_complex[k] *= tmp_buffer[k];
    //   // fft_complex[k] *= tmp_buffer[k] * std::exp(im*static_cast<cfloat>(rgen.sum3rand(2.0)));
    // }

    // do_pitch_shift(fft_complex, spectrum, tmp_buffer, new_freq/fund_freq, FFT_COMPLEX_SIZE);

    psola_analysis();

    // memcpy(fft_complex, spectrum, FFT_COMPLEX_SIZE * sizeof(cfloat));
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

    // Move the last part to the beginning
    memmove(in_buffer, in_buffer + HOP_SIZE, SHUNT_SIZE * sizeof(float));
  }

  memcpy(out, out_buffer + pos, inNumSamples * sizeof(float));
  // readpos += inNumSamples;


  // if (readpos >= FFT_SIZE) {
  //   readpos = 0;
  // }

  unit->pos = pos;
  unit->fund_freq = fund_freq;
}

void Autotune_Dtor(Autotune *unit) {
  fftwf_destroy_plan(unit->fft);
  fftwf_destroy_plan(unit->ifft);

  RTFree(unit->mWorld, unit->win);
  RTFree(unit->mWorld, unit->in_buffer);
  RTFree(unit->mWorld, unit->out_buffer);
  RTFree(unit->mWorld, unit->tmp_buffer);
  RTFree(unit->mWorld, unit->phase_buffer);
  RTFree(unit->mWorld, unit->fft_real);
  RTFree(unit->mWorld, unit->fft_complex);
  RTFree(unit->mWorld, unit->spectrum);
}

PluginLoad(Autotune) {
  ft = inTable;
  DefineDtorUnit(Autotune);
}
