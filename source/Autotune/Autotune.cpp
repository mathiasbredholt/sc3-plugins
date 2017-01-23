// Copyright 2017 M. Bredholt and M. Kirkegaard
// requires libsamplerate
#include "SC_PlugIn.h"
#include <samplerate.h>
#include <fftw3.h>
#include <complex>
#include <climits>
#include <fstream>
#include <string>


#define FFT_SIZE 4096
#define FFT_COMPLEX_SIZE static_cast<int>(FFT_SIZE/2 + 1)
#define WIN_SIZE 2048
#define HOP_SIZE 512
#define WRITEPOS 6144
#define SHUNT_SIZE (FFT_SIZE - HOP_SIZE)
#define PI 3.1415926535898f
#define TWOPI 6.28318530717952646f
#define EXPECTED_PHASE_DIFF (TWOPI * HOP_SIZE/WIN_SIZE)
#define CORRELATION_CONSTANT 0.9
#define MINIMUM_FREQUENCY 70
#define MAXIMUM_FREQUENCY 550
#define DEFAULT_PERIOD 441
#define DOWNSAMPLING 4


#define HANN_WINDOW 0
#define SINE_WINDOW 1
#define CEPSTRUM_LOWPASS 2

#define CEPSTRUM_CUTOFF static_cast<int>(0.1 * WIN_SIZE)

static InterfaceTable *ft;

float *debug;

typedef std::complex<float> cfloat;

const cfloat im(0.0, 1.0);

const float freq_grid[] = {
  130.81, 155.56, 174.61, 196.00, 233.08,
  261.63, 311.13, 349.23, 392.00, 466.16,
  523.25, 622.25, 698.46, 783.99, 923.33
};

float* alloc_buffer(int N, Unit *unit) {
  return static_cast<float*>(RTAlloc(unit->mWorld, N * sizeof(float)));
}

int* alloc_int_buffer(int N, Unit *unit) {
  return static_cast<int*>(RTAlloc(unit->mWorld, N * sizeof(int)));
}

cfloat* alloc_complex_buffer(int N, Unit *unit) {
  return static_cast<cfloat*>(RTAlloc(unit->mWorld, N * sizeof(cfloat)));
}

void clear_buffer(float *buffer, int N) {
  memset(buffer, 0, N * sizeof(float));
}

void clear_complex_buffer(cfloat *buffer, int N) {
  memset(buffer, 0, N * sizeof(cfloat));
}

void do_windowing(float *in, int wintype, int N, float scale = 1.0) {
  if (wintype == HANN_WINDOW) {
    for (int i = 0; i <   N; ++i) {
      // in[i] *= pow((0.5 - 0.5 * cos(TWOPI * i / (N - 1))), 2) * scale;
      in[i] *= (0.5 - 0.5 * cos(TWOPI * i / (N - 1))) * scale;
    }
  } else if (wintype == SINE_WINDOW) {
    for (int i = 0; i < N; ++i) {
      in[i] *= sin(TWOPI * i / N) * scale;
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

void do_exp_real(float *out, cfloat *in, int N) {
  for (int i = 0; i < N; ++i) {
    out[i] = exp(std::real(in[i]));
  }
}

void do_zeropad(float *in, int old_length, int new_length) {
  memset(in + old_length, 0, (new_length - old_length) * sizeof(float));
}

void do_fft(cfloat *fft_complex, float *fft_real, int N) {
  fftwf_plan fft;
  fft = fftwf_plan_dft_r2c_1d(
          N,
          fft_real,
          reinterpret_cast<fftwf_complex*>(fft_complex),
          FFTW_ESTIMATE);
  fftwf_execute(fft);
  fftwf_destroy_plan(fft);
}

void do_ifft(float *fft_real, cfloat *fft_complex, int N) {
  fftwf_plan ifft;
  ifft = fftwf_plan_dft_c2r_1d(
           N,
           reinterpret_cast<fftwf_complex*>(fft_complex),
           fft_real,
           FFTW_ESTIMATE);
  fftwf_execute(ifft);
  fftwf_destroy_plan(ifft);
}

void do_resample(float *out_buffer, float *in_buffer, float ratio,
                 int input_frames, int output_frames, int conversion_type = SRC_SINC_FASTEST) {
  int err;
  SRC_STATE *src;
  SRC_DATA src_data;
  src = src_new(conversion_type, 1, &err);
  src_data.data_in = in_buffer;
  src_data.data_out = out_buffer;
  src_data.input_frames = input_frames;
  src_data.output_frames = output_frames;
  src_data.src_ratio = ratio;
  src_data.end_of_input = 1;
  src_process(src, &src_data);
  src_delete(src);
}

float quad_fit_peak(int x2, float y1, float y2, float y3) {
  float top_x, top_y;
  top_x = ((2.0 * y1 - 4.0 * y2 + 2.0 * y3) * x2 + y1 - y3) / (2.0 *  y1 - 4.0 * y2 + 2.0 * y3);
  // top_y = (- pow(y1, 2.0) + (8.0 * y2 + 2.0 * y3) * y1 - 16.0 * pow((y2 - 1.0 / 4.0 * y3), 2.0)) / (8.0 * y1 - 16.0 * y2 + 8.0 * y3);
  return top_x;
}

float calc_signal_energy(float *in_buffer, int N) {
  float energy = 0.0;
  for (int i = 0; i < N; ++i) {
    energy += in_buffer[i] * in_buffer[i];
  }
  return energy;
}

int do_pitch_track2(float *x, float *resampled, float *NSDF, float sample_rate, int N) {
  int lagmin = static_cast<int>(sample_rate / (DOWNSAMPLING * MAXIMUM_FREQUENCY));
  int lagmax = static_cast<int>(sample_rate / (DOWNSAMPLING * MINIMUM_FREQUENCY));
  int idx_max = 0;
  int M = lagmax - lagmin + 1;
  float NSDF_max = 0.0;
  int period;

  memset(NSDF, 0, N * sizeof(float));

  do_resample(resampled, x, 1.0 / DOWNSAMPLING, N, N / DOWNSAMPLING, SRC_LINEAR);
  x = resampled;

  for (int i = lagmin; i < lagmax; ++i) {
    float x1_sq = 0.0, x2_sq = 0.0, acf = 0.0;
    for (int j = 0; j < M; ++j) {
      float x1 = x[j], x2 = x[i + j];
      x1_sq += x1 * x1;
      x2_sq += x2 * x2;
      acf += x1 * x2;
    }
    NSDF[i - lagmin] = acf / sqrt(x1_sq * x2_sq);
  }

  // Find maximum
  for (int i = 0; i < M; ++i) {
    if (NSDF[i] > NSDF_max) {
      NSDF_max = NSDF[i];
      idx_max = i;
    }
  }

  // No usable peak is found
  if (NSDF_max < 0.8) {
    return DEFAULT_PERIOD;
    // Search for peak
  } else {
    for (int i = 0; i < M; ++i) {
      if (NSDF[i + 1] > NSDF_max * CORRELATION_CONSTANT) {
        while (NSDF[i + 1] > NSDF[i]) {
          i++;
        }
        return quad_fit_peak((i + lagmin) * DOWNSAMPLING, NSDF[i - 1], NSDF[i], NSDF[i + 1]);
      } else {
        period = quad_fit_peak((idx_max + lagmin) * DOWNSAMPLING, NSDF[idx_max - 1], NSDF[idx_max], NSDF[idx_max + 1]);
      }
    }
  }
  return period;
}

// float do_autocorrelation(cfloat *in, float *phase_buffer,
//                          float sample_rate, int N) {
//   float highest_peak = -1e9;
//   int index = 0;
//   float trueFreq = 0;
//   float diff;
//   int tmp;
//   float freq_per_bin = sample_rate / (2.0 * N);

//   for (int i = 0; i < N; ++i) {
//     // Phase calculations
//     diff = std::arg(in[i]) - phase_buffer[i];
//     phase_buffer[i] = std::arg(in[i]);
//     diff -= static_cast<float>(i) * EXPECTED_PHASE_DIFF;

//     tmp = diff / PI;
//     if (tmp >= 0) {
//       tmp += tmp & 1;
//     } else {
//       tmp -= tmp & 1;
//     }
//     diff -= PI * tmp;

//     /* get deviation from bin frequency from the +/- Pi interval */
//     diff = static_cast<float>(HOP_SIZE) / WIN_SIZE * diff / TWOPI;

//     /* compute the k-th partials' true frequency */
//     diff = static_cast<float>(i) * freq_per_bin + diff * freq_per_bin;

//     float corr = std::real(in[i] * std::conj(in[i]));
//     if (corr > highest_peak) {
//       highest_peak = corr;
//       index = i;
//       trueFreq = diff;
//     }
//   }
//   return trueFreq;
// }

// void calc_power_spectral_density(cfloat *spectrum, int N) {
//   for (int i = 0; i < N; ++i) {
//     spectrum[i] *= std::conj(spectrum[i]);
//   }
// }



// void calc_square_difference_function(float *autocorrelation,
//                                      float *in_buffer, int N) {
//   float sum_squared = autocorrelation[0];
//   float sum_sq_left = sum_squared, sum_sq_right = sum_squared;

//   for (int i = 0; i < N / 2; ++i) {
//     sum_sq_left  -= pow(in_buffer[i], 2);
//     sum_sq_right -= pow(in_buffer[N - 1 - i], 2);
//     autocorrelation[i] *= 2.0 / (sum_sq_left + sum_sq_right);
//     debug[i] = autocorrelation[i];
//   }

// }

// void do_peak_picking(float *sdf, float sample_rate, int N, int *period, float *corr) {
//   int i = 0;
//   int n = 0;
//   int peaks[64];
//   int local_max;
//   int global_max;
//   float top_x;
//   float top_y;

//   int x1, x2, x3;
//   float y1, y2, y3;

//   while (n < 64 && i < N / 2 - 1) {
//     // Find positively sloped zero-crossing
//     if (sdf[i] < 0 && sdf[i + 1] >= 0) {
//       local_max = i;
//       // Find largest local maxima
//       while (n < 64 && i < N / 2 - 1) {
//         // Current value is a local maxima within zero-crossings
//         if (sdf[i] > sdf[local_max] && sdf[i] > sdf[i - 1] && sdf[i] > sdf[i + 1]) {
//           local_max = i;
//         }
//         // Negatively sloped zero-crossing detected - add peak to list
//         if (sdf[i] > 0 && sdf[i + 1] <= 0) {
//           peaks[n] = local_max;
//           n++;
//           break;
//         }
//         i++;
//       }
//       // No zero-crossing was found - save last peak
//       if (i == N / 2 - 1) {
//         peaks[n] = local_max;
//         n++;
//         break;
//       }
//     }
//     i++;
//   }

//   if (n == 0) {
//     *period = DEFAULT_PERIOD;
//     *corr = 0.0;
//     return;
//   }


//   // Find global max
//   global_max = peaks[0];

//   for (int j = 1; j < n - 1; ++j) {
//     if (sdf[peaks[j]] > sdf[global_max]) {
//       global_max = peaks[j];
//     }
//   }

//   // Find first peak above thresh
//   for (i = 0; i < n - 1; ++i) {
//     if (sdf[peaks[i]] > sdf[global_max] * CORRELATION_CONSTANT) {
//       break;
//     }
//   }

//   x1 = peaks[i] - 1;
//   x2 = peaks[i];
//   x3 = peaks[i] + 1;
//   y1 = sdf[x1];
//   y2 = sdf[x2];
//   y3 = sdf[x3];

//   top_x = ((2.0 * y1 - 4.0 * y2 + 2.0 * y3) * x2 + y1 - y3) / (2.0 *  y1 - 4.0 * y2 + 2.0 * y3);
//   top_y = (- pow(y1, 2.0) + (8.0 * y2 + 2.0 * y3) * y1 - 16.0 * pow((y2 - 1.0 / 4.0 * y3), 2.0)) / (8.0 * y1 - 16.0 * y2 + 8.0 * y3);

//   if (top_y > 0.2 && MAXIMUM_FREQUENCY > sample_rate / top_x && MINIMUM_FREQUENCY <    sample_rate / top_x) {
//     *period = static_cast<int>(round(top_x * 8));
//   } else {
//     *period = DEFAULT_PERIOD;
//   }

//   *corr = top_y;
// }

float closest_period(int period, float sample_rate) {
  int k = 0;
  float diff = 1e9;
  for (int i = 0; i < 15; ++i) {
    if (abs(period - (sample_rate / freq_grid[i])) < diff) {
      diff = abs(period - (sample_rate / freq_grid[i]));
      k = i;
    }
  }
  return (sample_rate / freq_grid[k]);
}

// void pitch_tracking(float *in_buffer, float *fft_real, cfloat *fft_complex, int *period, float *corr, float *energy, float sample_rate, int N) {
//   int win_size = N / 8;
//   int half_win_size = N / 16;
//   do_resample(fft_real, in_buffer, 0.125, FFT_SIZE, win_size, SRC_LINEAR);
//   *energy = calc_signal_energy(fft_real, win_size);
//   // if (energy > 0.4) {
//   do_fft(fft_complex, fft_real, win_size);
//   calc_power_spectral_density(fft_complex, half_win_size);
//   do_ifft(fft_real, fft_complex, win_size);
//   calc_square_difference_function(fft_real, in_buffer, half_win_size);
//   do_peak_picking(fft_real, sample_rate / 8, half_win_size, period, corr);
//   // } else {
//   // *period = DEFAULT_PERIOD;
//   // }
// }

// int create_pulse_train(float *in, float freq, float sample_rate,
//                        int N, int current_offset) {
//   int period = static_cast<int>(sample_rate / freq + 0.5);
//   int repeats = static_cast<int>((N - current_offset) / period);
//   int new_offset = (repeats + 1) * period + current_offset - N;

//   memset(in, 0, N * sizeof(float));

//   in[current_offset] = 1.0;
//   for (int i = current_offset + 1; i - current_offset < period; ++i) {
//     in[i] = 0.0;
//   }

//   for (int i = 0; i < repeats; ++i) {
//     memcpy(in + i * period, in, period * sizeof(float));
//   }

//   return new_offset;
// }

// void do_pitch_shift(cfloat *new_spectrum, cfloat *orig_spectrum,
//                     float *spectral_env, float ratio, int N) {
//   int new_index;
//   memset(new_spectrum, 0, N * sizeof(cfloat));
//   for (int i = 0; i < N; ++i) {
//     new_index = static_cast<int>(i * ratio);
//     if (new_index >= N) break;
//     // new_spectrum[new_index] += abs(orig_spectrum[i]);
//     new_spectrum[new_index] += abs(orig_spectrum[i]) *
//                                // std::exp(im * std::arg(orig_spectrum[i])) *
//                                static_cast<cfloat>(spectral_env[new_index] / spectral_env[i]);
//   }
// }

// void psola(float *out_buffer, float *in_buffer, float *tmp_buffer,
//            float fund_freq, float new_freq,
//            float sample_rate, int N) {
//   int k, j;
//   float pitch_scale = new_freq / fund_freq;
//   int period = static_cast<int>(sample_rate/fund_freq);

//   memset(out_buffer, 0, FFT_SIZE * sizeof(float));

//   // Create Hann window
//   memset(tmp_buffer, 0, 2 * period * sizeof(float));
//   for (int i = 0; i < 2 * period; ++i) {
//     tmp_buffer[i] = 1.0;
//   }
//   do_windowing(tmp_buffer, HANN_WINDOW, 2 * period, 1.0);

//   k = period;
//   j = static_cast<int>(period * pitch_scale);

//   while (k < N) {
//     if (k == 0) {
//       for (int i = 0; i < 2 * period; ++i) {
//         if (i < period) {
//           out_buffer[j + i] += in_buffer[k + i];
//         } else {
//           out_buffer[j + i] += in_buffer[k + i] * tmp_buffer[i];
//         }
//       }
//     } else {
//       for (int i = 0; i < 2 * period; ++i) {
//         out_buffer[j + i] += in_buffer[k + i] * tmp_buffer[i];
//       }
//     }

//     k += period;
//     j += static_cast<int>(period * pitch_scale);
//   }
// }

// void antialias(float *out_buffer, float *in_buffer, int N) {
//   float B[] = { 0.2929, 0.5858, 0.2929 };
//   float A[] = { 1, -2.6368e-16, 0.1716 };
//   memset(out_buffer, 0, N * sizeof(float));
//   for (int i = 0; i < N; ++i) {
//     if (i == 0) {
//       out_buffer[i] = B[0] * in_buffer[i]
//                       - A[0] * out_buffer[i];
//     } else if (i == 1) {
//       out_buffer[i] = B[1] * in_buffer[i - 1] + B[0] * in_buffer[i]
//                       - (A[1] * out_buffer[i - 1] + A[0] * out_buffer[i]);
//     } else {
//       out_buffer[i] = B[2] * in_buffer[i - 2] + B[1] * in_buffer[i - 1] + B[0] * in_buffer[i]
//                       - (A[2] * out_buffer[i - 2] + A[1] * out_buffer[i - 1] + A[0] * out_buffer[i]);
//     }

//   }
// }

float do_moving_average(float in, float *in_buffer, int N) {
  float out = 0.0;
  memmove(in_buffer + 1, in_buffer, (N - 1) * sizeof(float));
  in_buffer[0] = in;
  for (int i = 0; i < N; ++i) {
    out += in_buffer[i] / static_cast<float>(N);
  }
  return out;
}

struct Autotune : public Unit {
  float *in_buffer, *out_buffer, *fft_real, *tmp_buffer, *correlation_buffer, *resampling_buffer;
  float *freq_buffer;
  cfloat *fft_complex;
  SndBuf *m_buf;
  float m_fbufnum;
  int pos;
  int oscillator_offset;
  float average_freq;
  int period;
  int *period_buffer, *marks_buffer;
  int period_count;
  float tk;
  int readpos;
  bool read_flag;
};

extern "C" {
  void Autotune_next(Autotune *unit, int inNumSamples);
  void Autotune_Ctor(Autotune *unit);
  void Autotune_Dtor(Autotune *unit);
}

void Autotune_Ctor(Autotune *unit) {
  unit->in_buffer = alloc_buffer(FFT_SIZE, unit);
  unit->out_buffer = alloc_buffer(FFT_SIZE * 2, unit);
  unit->fft_real = alloc_buffer(FFT_SIZE, unit);
  unit->tmp_buffer = alloc_buffer(FFT_SIZE * 2, unit);
  unit->correlation_buffer = alloc_buffer(FFT_SIZE, unit);
  unit->resampling_buffer = alloc_buffer(FFT_SIZE, unit);
  // unit->fft_complex = alloc_complex_buffer(FFT_COMPLEX_SIZE, unit);

  clear_buffer(unit->in_buffer, FFT_SIZE);
  clear_buffer(unit->out_buffer, FFT_SIZE * 2);
  clear_buffer(unit->fft_real, FFT_SIZE);
  clear_buffer(unit->tmp_buffer, FFT_SIZE * 2);
  clear_buffer(unit->correlation_buffer, FFT_SIZE);
  clear_buffer(unit->resampling_buffer, FFT_SIZE);
  // clear_complex_buffer(unit->fft_complex, FFT_COMPLEX_SIZE);


  unit->pos = 0;
  // unit->oscillator_offset = 0;
  unit->freq_buffer = alloc_buffer(8, unit);
  for (int i = 0; i < 8; ++i) {
    unit->freq_buffer[i] = DEFAULT_PERIOD;
  }

  unit->m_fbufnum = -1e-9;

  unit->period = WIN_SIZE;
  unit->period_buffer = alloc_int_buffer(8, unit);
  unit->marks_buffer = alloc_int_buffer(8, unit);
  unit->period_count = 0;
  unit->tk = unit->period;
  // memset(unit->period_buffer, 0, 8 * sizeof(int));
  // memset(unit->marks_buffer, 0, 8 * sizeof(int));

  for (int i = 0; i < 8; ++i) {
    unit->period_buffer[i] = 0;
    unit->marks_buffer[i] = 0;
  }

  unit->period_buffer[0] = unit->period;

  unit->readpos = 0;

  unit->read_flag = false;

  SETCALC(Autotune_next);
  Autotune_next(unit, 1);
}

// void Autotune_next(Autotune *unit, int inNumSamples) {
//   float *in = IN(1);
//   float *out = OUT(0);
//   int pos = unit->pos;

//   float *in_buffer = unit->in_buffer;
//   float *out_buffer = unit->out_buffer;
//   float *tmp_buffer = unit->tmp_buffer;
//   float *resampler_in = unit->resampler_in;
//   float *fft_real = unit->fft_real;
//   cfloat *fft_complex = unit->fft_complex;
//   float *freq_buffer = unit->freq_buffer;

//   float average_freq = unit->average_freq;
//   float new_freq = 1;
//   float corr = 0.0;

//   RGen &rgen = *unit->mParent->mRGen;

//   GET_BUF

//   memcpy(in_buffer + SHUNT_SIZE + pos, in, inNumSamples * sizeof(float));
//   pos += inNumSamples;

//   if (pos >= HOP_SIZE) {
//     float f0_freq = 440.0;
//     pos = 0;

//     // ++++ Pitch tracking ++++
//     pitch_tracking(in_buffer, fft_real, fft_complex, &f0_freq, &corr, SAMPLERATE);

//     freq_buffer[f0_count] = f0_freq;
// f0_count++;
// f0_count &= 0x3; // 4 counter

// average_freq = do_moving_average(freq_buffer, 4);
// new_freq = closest_frequency(average_freq * 2);

//     // ++++++++++++++++++++++

//     psola(resampler_in, in_buffer, tmp_buffer, average_freq, new_freq, SAMPLERATE, WIN_SIZE);
//     // memcpy(bufData, fft_real, FFT_SIZE * sizeof(float));

//     printf("%f\n", new_freq/average_freq);

//     clear_buffer(tmp_buffer, FFT_SIZE);
//     do_resample(tmp_buffer, resampler_in, average_freq/new_freq,
//                 static_cast<int>(WIN_SIZE * new_freq/average_freq),
//                 WIN_SIZE, SRC_SINC_FASTEST);

//     // do_windowing(fft_real, HANN_WINDOW, FFT_SIZE, 1.0);

//     // printf("%d, %d\n", resampler_data.input_frames_used, resampler_data.output_frames_gen);


//     // ++++ Spectral envelope estimation ++++

//     // Create cepstrum
//     // do_fft(fft_complex, in_buffer, WIN_SIZE);
//     // do_log_abs(fft_complex, WIN_SIZE / 2);
//     // do_ifft(fft_real, fft_complex, WIN_SIZE);

//     // Window cepstrum to estimate spectral envelope
//     // do_windowing(fft_real, CEPSTRUM_LOWPASS, WIN_SIZE, 1.0 / WIN_SIZE);

//     // Go back to frequency domain to prepare for filtering
//     // do_fft(fft_complex, fft_real, WIN_SIZE);
//     // do_exp_real(fft_real, fft_complex, WIN_SIZE / 2);
//     // +++++++++++++++++++++++++++++++++++++++

//     // Create excitation signal
//     // unit->oscillator_offset = create_pulse_train(fft_real, new_freq, SAMPLERATE,
//     // WIN_SIZE, unit->oscillator_offset);


//     // Circular convolution
//     // do_fft(fft_complex, tmp_buffer, WIN_SIZE);
//     // for (int k = 0; k < FFT_COMPLEX_SIZE; ++k) {
//     //   fft_complex[k] *= fft_real[k];
//     //   // fft_complex[k] *= tmp_buffer[k] * std::exp(im*static_cast<cfloat>(rgen.sum3rand(2.0)));
//     // }

//     // do_ifft(fft_real, fft_complex, WIN_SIZE);
//     // do_windowing(fft_real, HANN_WINDOW, WIN_SIZE, 1.0/WIN_SIZE);


//     // !!! SILENCE !!!
//     // clear_buffer(fft_real, FFT_SIZE);

//     memmove(out_buffer, out_buffer + HOP_SIZE, SHUNT_SIZE * sizeof(float));

//     for (int i = 0; i < FFT_SIZE; ++i) {
//       if (i < SHUNT_SIZE) {
//         // Overlap and add
//         out_buffer[i] += fft_real[i];
//       } else {
//         // Copy
//         out_buffer[i] = fft_real[i];
//       }
//     }

//     // Move the last part to the beginning
//     memmove(in_buffer, in_buffer + HOP_SIZE, SHUNT_SIZE * sizeof(float));

//   }

//   memcpy(out, out_buffer + pos, inNumSamples * sizeof(float));
//   unit->pos = pos;
//   unit->average_freq = average_freq;
//   unit->f0_count = f0_count;
// }

void Autotune_next(Autotune *unit, int inNumSamples) {
  float *in = IN(1);
  float *out = OUT(0);
  int pos = unit->pos;
  int readpos = unit->readpos;

  float *in_buffer = unit->in_buffer;
  float *out_buffer = unit->out_buffer;
  float *tmp_buffer = unit->tmp_buffer;
  float *correlation_buffer = unit->correlation_buffer;
  float *resampling_buffer = unit->resampling_buffer;
  float *fft_real = unit->fft_real;
  // cfloat *fft_complex = unit->fft_complex;

  int period = unit->period;
  int shunt_size = WIN_SIZE - period;


  int old_period;
  int *period_buffer = unit->period_buffer;
  int *marks_buffer = unit->marks_buffer;
  int period_count = unit->period_count;

  float tk = unit->tk;

  GET_BUF

  memcpy(in_buffer + pos, in, inNumSamples * sizeof(float));
  pos += inNumSamples;

  if (pos - shunt_size >= period) {
    // Step 1.0 --- PITCH TRACKING
    memcpy(resampling_buffer, in_buffer, FFT_SIZE * sizeof(float));
    // period = do_pitch_track2(resampling_buffer, fft_real, correlation_buffer, SAMPLERATE, WIN_SIZE);
    period = 441;


    period_count++;
    period_buffer[period_count] = period;

    for (int i = 0; i < period_count + 1; ++i) {
      if (i == 0) {
        marks_buffer[i] = period_buffer[i];
      } else {
        marks_buffer[i] = period_buffer[i] + marks_buffer[i - 1];
      }
    }

    memcpy(tmp_buffer + marks_buffer[period_count - 1] * 2,
           in_buffer,
           2 * period_buffer[period_count] * sizeof(float));

    do_windowing(tmp_buffer + marks_buffer[period_count - 1] * 2,
                 HANN_WINDOW,
                 2 * period_buffer[period_count]);

    memcpy(bufData, tmp_buffer, FFT_SIZE * 2 * sizeof(float));

    // Step 5.0 -- MOVE INPUT BUFFER
    pos -= period;
    memmove(in_buffer, in_buffer + period, (FFT_SIZE - period) * sizeof(float));
  }

  while (period_count >= 2) {
    int min_t = INT_MAX;
    int odt, dt;
    int idx = 0;
    int old_period;
    int new_period;
    int offset;
    float new_pitchscale = 1.0, old_pitchscale = 1.0;

    // Step 2.0 ---- DECIDE INDEX (0, 1, 2)
    for (int i = 0; i < period_count + 1; ++i) {
      if (min_t > abs(marks_buffer[i] - tk)) {
        idx = i;
        min_t = abs(marks_buffer[i] - tk);
      }
    }

    // new_pitchscale =
    // fmin(2.0, fmax(0.75, static_cast<float>(new_period) /
    // closest_period(new_period, SAMPLERATE)));

    new_period = period_buffer[idx];
    dt = static_cast<int>(static_cast<float>(new_period) / new_pitchscale);

    if (idx == 0) {
      tk += dt;
      offset = 0;
      old_period = period_buffer[0];
    } else {
      tk += dt - marks_buffer[idx - 1];
      old_period = period_buffer[idx - 1];
      offset = marks_buffer[idx - 1] * 2;
    }

    // old_pitchscale =
    // fmin(2.0, fmax(0.75, static_cast<float>(old_period) /
    // closest_period(old_period, SAMPLERATE)));
    odt = static_cast<int>(static_cast<float>(old_period) / new_pitchscale);


    // Step 3.0 --- WRITE TO OUTPUT BUFFER
    // Move output buffer with period
    memmove(out_buffer, out_buffer + odt, (FFT_SIZE * 2 - odt) * sizeof(float));
    readpos -= odt;
    printf("%d \t %d\n", readpos, pos);


    clear_buffer(out_buffer + WRITEPOS, WIN_SIZE);

    for (int i = 0; i < 2 * new_period; ++i) {
      out_buffer[WRITEPOS - new_period + i] +=
        tmp_buffer[offset + i] / new_pitchscale;
      // tmp_buffer[i];
    }

    // Step 4.0 --- UPDATE
    memmove(period_buffer, period_buffer + idx, (8 - idx) * sizeof(int));

    memmove(tmp_buffer,
            tmp_buffer + offset,
            (FFT_SIZE * 2 - offset) * sizeof(float));

    period_count -= idx;
  }

  readpos += inNumSamples;
  memcpy(out, out_buffer + readpos, inNumSamples * sizeof(float));

  unit->period = period;
  unit->tk = tk;
  unit->pos = pos;
  unit->readpos = readpos;
  unit->period_count = period_count;
}


void Autotune_Dtor(Autotune * unit) {
  RTFree(unit->mWorld, unit->in_buffer);
  RTFree(unit->mWorld, unit->out_buffer);
  RTFree(unit->mWorld, unit->resampling_buffer);
  RTFree(unit->mWorld, unit->correlation_buffer);
  RTFree(unit->mWorld, unit->tmp_buffer);
  RTFree(unit->mWorld, unit->fft_real);
  // RTFree(unit->mWorld, unit->fft_complex);
  RTFree(unit->mWorld, unit->freq_buffer);
  RTFree(unit->mWorld, unit->period_buffer);
  RTFree(unit->mWorld, unit->marks_buffer);
}

PluginLoad(Autotune) {
  ft = inTable;
  DefineDtorUnit(Autotune);
}




// period_buffer[period_count] = period;
// period_count++;

// // Update marks buffer
// for (int i = 0; i < period_count; ++i) {
//   if (i == 0) {
//     marks_buffer[i] = period_buffer[period_count - 1];
//   } else {
//     marks_buffer[i] = marks_buffer[i - 1] + period_buffer[period_count - 1 - i];
//   }
// }

//   // // +++++++++++++ Analysis +++++++++++++
//   pitch_tracking(in_buffer, fft_real, fft_complex, &period, &corr, &energy, SAMPLERATE, FFT_SIZE);
// period = do_moving_average(tmp_period, unit->freq_buffer, 8);

// Save input and window
// memcpy(tmp_buffer, in_buffer, )
// memcpy(tmp_buffer + marks_buffer[period_count] - period_buffer[period_count],
//        in_buffer, 2 * period_buffer[period_count] * sizeof(float));

// do_windowing(tmp_buffer + marks_buffer[period_count] - period_buffer[period_count],
//              HANN_WINDOW,
//              2 * period_buffer[period_count]);


// // +++++++ Synthesis ++++++++++++++++
// while (period_count > 0) {
//   int min_t = INT_MAX;
//   int idx = 0;
//   int seg_len;
//   float pitchscale;
//   float dt;

//   seg_len = period_buffer[idx]; // 441

//   pitchscale = 1.0;

//   // 1166 + 2 * seg_len = 2048
//   memcpy(tmp_buffer, in_buffer + WIN_SIZE - marks_buffer[idx] - seg_len, 2 * seg_len * sizeof(float));
//   do_windowing(tmp_buffer, HANN_WINDOW, 2 * seg_len);

//   memmove(out_buffer, out_buffer + old_period, shunt_size * sizeof(float));

//   // Add to existing data
//   for (int j = 0; j < 2 * seg_len; ++j) {
//     out_buffer[WIN_SIZE - marks_buffer[idx] - seg_len + j] +=
//       tmp_buffer[j] / pitchscale;
//   }
// }
//   int min_t = INT_MAX;
//   int idx = 0;
//   int seg_len;
//   float pitchscale;
//   float dt;

// for (int i = 0; i < period_count; ++i) {
//   if (min_t > abs(marks_buffer[i + 1] - tk)) {
//     idx = i;
//     min_t = abs(marks_buffer[i + 1] - tk);
//   }
// }

//   seg_len = period_buffer[idx];

//   //   // if (correlation_buffer[idx] > 0.4) {    // Segment is voiced
//   //   // Find closest period in grid
//   pitchscale =  1.0;
//   // pitchscale = fmin(2.0, fmax(0.5,
//   // static_cast<float>(seg_len) / closest_period(seg_len, SAMPLERATE)));
//   //   // } else {                                // Segment is unvoiced
//   //   // seg_len = DEFAULT_PERIOD;
//   //   // idx = 1; // Go to next
//   //   // pitchscale = 1.0;
//   //   // }


//   // Time step
//   dt = static_cast<float>(seg_len) / pitchscale;


//   // Move output buffer with time step
//   memmove(out_buffer,
//           out_buffer + static_cast<int>(round(dt)),
//           (WIN_SIZE - static_cast<int>(round(dt))) * sizeof(float));

//   // +++++++++++ OVERLAP AND ADD +++++++++++++++++++
//   // ----64-----128-----192------256
//   // 0 --64 ----128
//   //     64 ----128 ----192------
//   //            128 ----192------256
//   clear_buffer(out_buffer + (WIN_SIZE - static_cast<int>(round(dt))), static_cast<int>(round(dt)));

//   memcpy(tmp_buffer, in_buffer + marks_buffer[idx] - seg_len, 2 * seg_len * sizeof(float));
//   do_windowing(tmp_buffer, HANN_WINDOW, 2 * seg_len);

//   // Add to existing data
//   for (int j = 0; j < 2 * seg_len; ++j) {
//     out_buffer[marks_buffer[idx] - seg_len + j] +=
//       // tmp_buffer[j] / pitchscale;
//       in_buffer[j];
//   }
//   // +++++++++++++++++++++++++++++++++++++++++++++++

//   // Increment time offset
//   // tk += dt - marks_buffer[idx];

//   // Remove segment from buffers
//   // memmove(tmp_buffer, tmp_buffer + marks_buffer[idx] - seg_len, (FFT_SIZE - 2 * seg_len) * sizeof(float));
//   // memmove(period_buffer, period_buffer + idx, period_count * sizeof(int));
//   memmove(period_buffer, period_buffer + period_count, period_count * sizeof(int));
//   period_count -= 1;

//   readpos -= static_cast<int>(round(dt));
//   unit->read_flag = true;
// }

// Move the last part to the beginning

// printf("%d\n", pos);

// if (unit->read_flag) {
