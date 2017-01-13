// Copyright 2017 M. Bredholt and M. Kirkegaard
#include "SC_PlugIn.h"
#include <samplerate.h>
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
#define MINIMUM_FREQUENCY 70
#define MAXIMUM_FREQUENCY 1024

#define HANN_WINDOW 0
#define SINE_WINDOW 1
#define CEPSTRUM_LOWPASS 2

#define CEPSTRUM_CUTOFF static_cast<int>(0.05 * WIN_SIZE)

static InterfaceTable *ft;

typedef std::complex<float> cfloat;


const cfloat im(0.0, 1.0);

const float freq_grid[] = {
  130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
  261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88
};

void do_windowing(float *in, int wintype, int N, float scale = 1.0) {
  if (wintype == HANN_WINDOW) {
    // Hann window squared
    for (int i = 0; i <   N; ++i) {
      in[i] *= pow((0.5 - 0.5 * cos(TWOPI*i/(N-1))), 2) * scale;
    }
  } else if (wintype == SINE_WINDOW) {
    for (int i = 0; i < N; ++i) {
      in[i] *= sin(TWOPI*i/N) * scale;
    }
  } else if (wintype == CEPSTRUM_LOWPASS) {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == CEPSTRUM_CUTOFF) {
        in[i] *= 1.0 * scale;
      } else if (1 <= i && i < CEPSTRUM_CUTOFF) {
        in[i] *= 2.0 * scale;
      } else if (CEPSTRUM_CUTOFF < i && i <= N - 1) {
        in[i] *= 0.0 * scale;
      }
    }
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
  float top_x;
  float top_y;

  int x1, x2, x3;
  float y1, y2, y3;

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

  x1 = peaks[i] - 1;
  x2 = peaks[i];
  x3 = peaks[i] + 1;
  y1 = sdf[x1];
  y2 = sdf[x2];
  y3 = sdf[x3];

  top_x = ((2.0 * y1 - 4.0 * y2 + 2.0 * y3) * x2 + y1 - y3) / (2.0 *  y1 - 4.0 * y2 + 2.0 * y3);
  top_y = (- pow(y1, 2.0) + (8.0 * y2 + 2.0 * y3) * y1 - 16.0 * pow((y2 - 1.0/4.0 * y3), 2.0))/(8.0 * y1 - 16.0 * y2 + 8.0 * y3);


  *freq = sample_rate/top_x;
  *corr = top_y;

  // *corr = sdf[peaks[i]];
  // if (*corr > 0.5)
    // *freq = sample_rate / static_cast<float>(peaks[i]);
}

float closest_frequency(float freq) {
  int k = 0;
  float diff = 1e6;
  for (int i = 0; i < 14; ++i) {
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

void psola(float *out_buffer, float *in_buffer, float *tmp_buffer, float *pitch_marks, float fund_freq, float new_freq, float sample_rate, int N) {
  float mod_rate = fmin(fmax(fund_freq/new_freq, 0.5), 1.5);
  int win_min = 2 * static_cast<int>(sample_rate/MINIMUM_FREQUENCY);
  int period = static_cast<int>(sample_rate/fund_freq);
  float offset = 0;
  int offset_idx = 0;
  int oa_idx = 0;
  int i = 1;

  memset(pitch_marks, 0, FFT_SIZE * sizeof(float));
  memset(out_buffer, 0, FFT_SIZE * sizeof(float));
  memset(tmp_buffer, 0, FFT_SIZE * sizeof(float));

  pitch_marks[0] = period;

  while (pitch_marks[i - 1] < N - win_min) {
    period = static_cast<int>(sample_rate/fund_freq);
    i++;
    pitch_marks[i] = period + pitch_marks[i - 1];
  }

  while (offset_idx < i - 2) {
    int period;
    offset += mod_rate;
    offset_idx = static_cast<int>(ceil(offset));
    period = static_cast<int>(pitch_marks[offset_idx + 1] - pitch_marks[offset_idx]);
    oa_idx += period;

    memcpy(tmp_buffer, in_buffer + offset_idx, 2 * period * sizeof(float));
    do_windowing(tmp_buffer, HANN_WINDOW, 2*period);

    for (int i = 0; i < 2*period; ++i) {
      out_buffer[oa_idx - period + i] += tmp_buffer[i]; 
    }
  }
}

void antialias(float *out_buffer, float *in_buffer, int N) {
  float B[] = { 0.2929, 0.5858, 0.2929 };
  float A[] = { 1, -2.6368e-16, 0.1716 };
  memset(out_buffer, 0, N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    if (i == 0) {
      out_buffer[i] = B[0] * in_buffer[i] 
      - A[0] * out_buffer[i];
    } else if (i == 1) {
      out_buffer[i] = B[1] * in_buffer[i - 1] + B[0] * in_buffer[i]
      - (A[1] * out_buffer[i - 1] + A[0] * out_buffer[i]);
    } else {
      out_buffer[i] = B[2] * in_buffer[i - 2] + B[1] * in_buffer[i - 1] + B[0] * in_buffer[i] 
      - (A[2] * out_buffer[i - 2] + A[1] * out_buffer[i - 1] + A[0] * out_buffer[i]);
    }

  }
}

void resample(float *out_buffer, float *in_buffer, float mod_rate, int N) {
  float M = mod_rate;
  float L = 1/mod_rate;

  for (int i = 0; i < N/64; ++i) {
    // Interpolation

  }

  memset(out_buffer, 0, N * sizeof(float));
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

  SRC_STATE* resampler;
  int resampler_err;
};

extern "C" {
  void Autotune_next(Autotune *unit, int inNumSamples);
  void Autotune_Ctor(Autotune *unit);
  void Autotune_Dtor(Autotune *unit);
}

void Autotune_Ctor(Autotune *unit) {
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

  unit->resampler = src_new(SRC_SINC_BEST_QUALITY, 1, &unit->resampler_err);

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
  cfloat *fft_complex = unit->fft_complex;
  cfloat *spectrum = unit->spectrum;
  int pos = unit->pos;
  float fund_freq = unit->fund_freq;
  float mod_rate = 1.0;
  float new_freq = 1;
  float corr = 0.0;


  SRC_STATE *resampler = unit->resampler;
  SRC_DATA *resampler_data;

  RGen &rgen = *unit->mParent->mRGen;

  GET_BUF

  memcpy(in_buffer + SHUNT_SIZE + pos, in, inNumSamples * sizeof(float));
  pos += inNumSamples;

  if (pos >= HOP_SIZE) {
    pos = 0;

    // FFT
    memcpy(fft_real, in_buffer, WIN_SIZE * sizeof(float));
    do_windowing(fft_real, HANN_WINDOW, WIN_SIZE);
    fftwf_execute(unit->fft);

    // // Save spectrum for later
    // memcpy(spectrum, fft_complex, FFT_COMPLEX_SIZE * sizeof(cfloat));

    // // ++++ Pitch tracking ++++
    // // fund_freq = do_autocorrelation(fft_complex, phase_buffer,
    //                                // sample_rate, FFT_COMPLEX_SIZE);

    calc_power_spectral_density(fft_complex, FFT_COMPLEX_SIZE);
    fftwf_execute(unit->ifft);
    calc_square_difference_function(fft_real, in_buffer, WIN_SIZE);
    do_peak_picking(fft_real, SAMPLERATE, WIN_SIZE, &fund_freq, &corr);

    new_freq = closest_frequency(fund_freq);

    // printf("%f\n", fund_freq);
    // printf("%f\n", new_freq/fund_freq);

    // +++++++++++++++++++++++

    mod_rate = fund_freq/new_freq;
    // psola(fft_real, in_buffer, tmp_buffer, phase_buffer, fund_freq, new_freq, SAMPLERATE, WIN_SIZE);
    // antialias(tmp_buffer, in_buffer, WIN_SIZE);

    
    resampler_data->data_in = in_buffer;
    resampler_data->data_out = fft_real;
    resampler_data->input_frames = WIN_SIZE;
    resampler_data->output_frames = FFT_SIZE;
    resampler_data->src_ratio = fund_freq/new_freq;
    resampler_data->end_of_input = 1;

    src_process(resampler, resampler_data);

    // resample(fft_real, tmp_buffer, mod_rate, WIN_SIZE);
    // memcpy(fft_real, tmp_buffer, WIN_SIZE * sizeof(float));
    // do_windowing(fft_real, HANN_WINDOW, WIN_SIZE, 1.0);

    // ++++ Spectral envelope estimation ++++

    // Create cepstrum
    // memcpy(fft_complex, spectrum, FFT_COMPLEX_SIZE * sizeof(cfloat));
    // do_log_abs(fft_complex, FFT_COMPLEX_SIZE);
    // fftwf_execute(unit->ifft);

    // // Window cepstrum to estimate spectral envelope
    // do_windowing(fft_real, CEPSTRUM_LOWPASS, FFT_SIZE, 1.0/FFT_SIZE);

    // // Go back to frequency domain to prepare for filtering
    // fftwf_execute(unit->fft);
    // do_real_exp(tmp_buffer, fft_complex, FFT_COMPLEX_SIZE);
    // // +++++++++++++++++++++++++++++++++++++++



    // // Create excitation signal
    // unit->oscillator_offset = create_pulse_train(fft_real, new_freq, SAMPLERATE,
    // WIN_SIZE, unit->oscillator_offset);

    // memcpy(bufData, fft_real, bufFrames * sizeof(float));

    // // Zeropad to match conditions for convolution
    // do_zeropad(fft_real, WIN_SIZE, FFT_SIZE);
    // fftwf_execute(unit->fft);
    // // Circular convolution
    // for (int k = 0; k < FFT_COMPLEX_SIZE; ++k) {
    //   

    // fft_complex[k] *= tmp_buffer[k];
    // //   // fft_complex[k] *= tmp_buffer[k] * std::exp(im*static_cast<cfloat>(rgen.sum3rand(2.0)));
    // // }

    // // do_pitch_shift(fft_complex, spectrum, tmp_buffer, new_freq/fund_freq, FFT_COMPLEX_SIZE);

    // memcpy(fft_complex, spectrum, FFT_COMPLEX_SIZE * sizeof(cfloat));
    // IFFT
    // fftwf_execute(unit->ifft);
    // do_windowing(fft_real, HANN_WINDOW, FFT_SIZE, 1.0/FFT_SIZE);

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

  src_delete(unit->resampler);
}

PluginLoad(Autotune) {
  ft = inTable;
  DefineDtorUnit(Autotune);
}
