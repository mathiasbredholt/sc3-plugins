// PitchCorrection UGen
// Copyright 2017 M. Bredholt and M. Kirkegaard
// requires libsamplerate
#include "SC_PlugIn.h"
#include <samplerate.h>
#include <climits>
#include <complex>

#define FFT_SIZE 4096
#define FFT_COMPLEX_SIZE static_cast<int>(FFT_SIZE/2 + 1)
#define WIN_SIZE 2048
#define WRITEPOS 6144
#define CORRELATION_CONSTANT 0.9
#define MINIMUM_FREQUENCY 70
#define MAXIMUM_FREQUENCY 680
#define DEFAULT_PERIOD 441
#define DOWNSAMPLING 4
#define TWOPI 6.28318530717952646f

#define HANN_WINDOW 0
#define SINE_WINDOW 1

static InterfaceTable *ft;

float* alloc_buffer(int N, Unit *unit) {
    return static_cast<float*>(RTAlloc(unit->mWorld, N * sizeof(float)));
}

int* alloc_int_buffer(int N, Unit *unit) {
    return static_cast<int*>(RTAlloc(unit->mWorld, N * sizeof(int)));
}

void clear_buffer(float *buffer, int N) {
    memset(buffer, 0, N * sizeof(float));
}

void calc_square_difference_function(float *ACF,
                                     float *x, int N) {
    float x_left = ACF[0], x_right = ACF[0];
    for (int i = 0; i < N / 2; ++i) {
        x_left  -= pow(x[i], 2);
        x_right -= pow(x[N - 1 - i], 2);
        ACF[i] *= 2.0 / (x_left + x_right);
    }
}

void do_windowing(float *in, int wintype, int N, float scale = 1.0) {
    if (wintype == HANN_WINDOW) {
        for (int i = 0; i <   N; ++i) {
            in[i] *= (0.5 - 0.5 * cos(TWOPI * i / (N - 1))) * scale;
        }
    } else if (wintype == SINE_WINDOW) {
        for (int i = 0; i < N; ++i) {
            in[i] *= sin(TWOPI * i / N) * scale;
        }
    }
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
    return ((2.0 * y1 - 4.0 * y2 + 2.0 * y3) * x2 + y1 - y3) / (2.0 *  y1 - 4.0 * y2 + 2.0 * y3);
}

float calc_signal_energy(float *in_buffer, int N) {
    float energy = 0.0;
    for (int i = 0; i < N; ++i) {
        energy += in_buffer[i] * in_buffer[i];
    }
    return energy;
}

int pitch_track(float *x, float *resampled, float *ACF, float sample_rate, int N) {
    int lagmin = static_cast<int>(sample_rate / (DOWNSAMPLING * MAXIMUM_FREQUENCY));
    int lagmax = static_cast<int>(sample_rate / (DOWNSAMPLING * MINIMUM_FREQUENCY));
    int idx_max = 0;
    int M = lagmax - lagmin + 1;
    // int M = N / 2;
    float ACF_max = 0.0;
    int P;
    float energy;
    float mean = 0.0;

    memset(ACF, 0, N * sizeof(float));

    do_resample(resampled, x, 1.0 / DOWNSAMPLING, N, N / DOWNSAMPLING, SRC_LINEAR);
    x = resampled;
    N = N / DOWNSAMPLING;

    energy = calc_signal_energy(x, N);
    if (energy < 0.5) return DEFAULT_PERIOD;

    // do_fft(fft_complex, x, FFT_SIZE);
    // calc_power_spectral_density(fft_complex, FFT_COMPLEX_SIZE);
    // do_ifft(ACF, fft_complex, FFT_SIZE);
    // calc_square_difference_function(ACF, x, N);

    for (int i = 0; i < N; ++i) {
        mean += x[i] / static_cast<float>(N);
    }

    for (int i = 0; i < N; ++i) {
        x[i] -= mean; // SUBTRACT THE MEAN
    }

    for (int i = lagmin; i < lagmax; ++i) {
        float x1_sq = 0.0, x2_sq = 0.0, acf = 0.0;
        for (int j = 0; j < M; ++j) {
            float x1 = x[j], x2 = x[i + j];
            x1_sq += x1 * x1;
            x2_sq += x2 * x2;
            acf += x1 * x2;
        }
        ACF[i - lagmin] = acf / sqrt(x1_sq * x2_sq);
    }

    // Find maximum
    for (int i = 0; i < M; ++i) {
        if (ACF[i] > ACF_max) {
            ACF_max = ACF[i];
            idx_max = i;
        }
    }

    // No usable peak is found
    if (ACF_max < 0.7) {
        return DEFAULT_PERIOD;
        // Search for peak
    } else {
        for (int i = 0; i < M; ++i) {
            if (ACF[i + 1] > ACF_max * CORRELATION_CONSTANT) {
                while (ACF[i + 1] > ACF[i]) {
                    i++;
                }
                return quad_fit_peak((i + lagmin) * DOWNSAMPLING, ACF[i - 1], ACF[i], ACF[i + 1]);
                // if (sample_rate / i > MINIMUM_FREQUENCY && sample_rate / i < MAXIMUM_FREQUENCY) {
                //     return quad_fit_peak(i * DOWNSAMPLING, ACF[i - 1], ACF[i], ACF[i + 1]);
                // } else {
                //     return DEFAULT_PERIOD;
                // }
            } else {
                P = quad_fit_peak((idx_max + lagmin) * DOWNSAMPLING, ACF[idx_max - 1], ACF[idx_max], ACF[idx_max + 1]);
                // P = quad_fit_peak(i * DOWNSAMPLING, ACF[idx_max - 1], ACF[idx_max], ACF[idx_max + 1]);
            }
        }
    }
    return P;
}

float closest_period(int P, float *scale_buf, int N, float sample_rate) {
    int k = 0;
    float diff = 1e9;
    for (int i = 0; i < N; ++i) {
        float p = sample_rate / scale_buf[i];
        if (abs(static_cast<float>(P) - p) < diff) {
            diff = abs(static_cast<float>(P) - p);
            k = i;
        }
    }
    return (sample_rate / scale_buf[k]);
}

float do_moving_average(float in, float *in_buffer, int N) {
    float out = 0.0;
    memmove(in_buffer + 1, in_buffer, (N - 1) * sizeof(float));
    in_buffer[0] = in;
    for (int i = 0; i < N; ++i) {
        out += in_buffer[i] / static_cast<float>(N);
    }
    return out;
}

struct PitchCorrection : public Unit {
    float *in_buffer, *out_buffer, *tmp_buffer, *segment_buffer, *correlation_buffer, *resampling_buffer;
    float *freq_buffer;
    SndBuf *m_buf;
    float m_fbufnum;
    int pos;
    int P;
    int *segment_lengths, *pitch_marks;
    int segments_ready;
    float T;
    int readpos;
    bool read_flag;
};

extern "C" {
    void PitchCorrection_next(PitchCorrection *unit, int inNumSamples);
    void PitchCorrection_Ctor(PitchCorrection *unit);
    void PitchCorrection_Dtor(PitchCorrection *unit);
}

void PitchCorrection_Ctor(PitchCorrection *unit) {
    unit->in_buffer = alloc_buffer(FFT_SIZE, unit);
    unit->out_buffer = alloc_buffer(FFT_SIZE * 2, unit);
    unit->tmp_buffer = alloc_buffer(FFT_SIZE, unit);
    unit->segment_buffer = alloc_buffer(FFT_SIZE * 2, unit);
    unit->correlation_buffer = alloc_buffer(FFT_SIZE, unit);
    unit->resampling_buffer = alloc_buffer(FFT_SIZE, unit);

    clear_buffer(unit->in_buffer, FFT_SIZE);
    clear_buffer(unit->out_buffer, FFT_SIZE * 2);
    clear_buffer(unit->tmp_buffer, FFT_SIZE);
    clear_buffer(unit->segment_buffer, FFT_SIZE * 2);
    clear_buffer(unit->correlation_buffer, FFT_SIZE);
    clear_buffer(unit->resampling_buffer, FFT_SIZE);

    unit->pos = 0;
    unit->freq_buffer = alloc_buffer(8, unit);
    for (int i = 0; i < 8; ++i) {
        unit->freq_buffer[i] = DEFAULT_PERIOD;
    }

    unit->m_fbufnum = -1e-9;

    unit->P = WIN_SIZE;
    unit->segment_lengths = alloc_int_buffer(8, unit);
    unit->pitch_marks = alloc_int_buffer(8, unit);
    unit->segments_ready = 0;
    unit->T = unit->P;

    for (int i = 0; i < 8; ++i) {
        unit->segment_lengths[i] = 0;
        unit->pitch_marks[i] = 0;
    }

    unit->segment_lengths[0] = unit->P;

    unit->readpos = 0;

    unit->read_flag = false;

    SETCALC(PitchCorrection_next);
    PitchCorrection_next(unit, 1);
}

void PitchCorrection_next(PitchCorrection *unit, int inNumSamples) {
    float *in = IN(1);
    float *out = OUT(0);
    int pos = unit->pos;
    int readpos = unit->readpos;

    float *in_buffer = unit->in_buffer;
    float *out_buffer = unit->out_buffer;
    float *segment_buffer = unit->segment_buffer;
    float *correlation_buffer = unit->correlation_buffer;
    float *resampling_buffer = unit->resampling_buffer;
    float *tmp_buffer = unit->tmp_buffer;

    int P = unit->P;
    int shunt_size = WIN_SIZE - P;


    int old_P;
    int *segment_lengths = unit->segment_lengths;
    int *pitch_marks = unit->pitch_marks;
    int segments_ready = unit->segments_ready;

    float T = unit->T;

    GET_BUF

    memcpy(in_buffer + pos, in, inNumSamples * sizeof(float));
    pos += inNumSamples;

    // printf("%d\n", pos);

    if (pos - shunt_size >= P) {
        int offset;
        // Step 1.0 --- PITCH TRACKING
        memcpy(resampling_buffer, in_buffer, FFT_SIZE * sizeof(float));
        P = pitch_track(resampling_buffer, tmp_buffer, correlation_buffer, SAMPLERATE, WIN_SIZE);

        segments_ready++;
        segment_lengths[segments_ready] = P;

        for (int i = 0; i < segments_ready + 1; ++i) {
            if (i == 0) {
                pitch_marks[i] = segment_lengths[i];
            } else {
                pitch_marks[i] = segment_lengths[i] + pitch_marks[i - 1];
            }
        }

        offset = pitch_marks[segments_ready - 1] * 2;

        memcpy(segment_buffer + offset,
               in_buffer,
               2 * segment_lengths[segments_ready] * sizeof(float));

        do_windowing(segment_buffer + offset,
                     HANN_WINDOW,
                     2 * segment_lengths[segments_ready]);

        // Step 5.0 -- MOVE INPUT BUFFER
        pos -= P;
        memmove(in_buffer, in_buffer + P, (FFT_SIZE - P) * sizeof(float));
    }

    while (segments_ready >= 2) {
        int min_t = INT_MAX;
        int odt, dt;
        int R = 0;
        int P0;
        int offset;
        float beta = 1.0;

        // Step 2.0 ---- DECIDE INDEX (0, 1, 2)
        for (int i = 0; i < segments_ready + 1; ++i) {
            if (min_t > abs(pitch_marks[i] - T)) {
                R = i;
                min_t = abs(pitch_marks[i] - T);
            }
        }

        P0 = segment_lengths[R];
        beta =
            fmin(2.0, fmax(0.5, static_cast<float>(P0) /
                           closest_period(P0, bufData, bufFrames, SAMPLERATE)));


        dt = static_cast<int>(static_cast<float>(P0) / beta);

        // If ind is zero repeat the same section
        if (R == 0) {
            T += dt;
            offset = 0;
        } else {
            T += dt - pitch_marks[R - 1];
            offset = pitch_marks[R - 1] * 2;
        }

        // Step 3.0 --- WRITE TO OUTPUT BUFFER
        // Move output buffer with P
        memmove(out_buffer, out_buffer + dt, (FFT_SIZE * 2 - dt) * sizeof(float));
        readpos -= dt;

        clear_buffer(out_buffer + WRITEPOS, WIN_SIZE);

        for (int i = 0; i < 2 * P0; ++i) {
            out_buffer[WRITEPOS - P0 + i] +=
                segment_buffer[offset + i] / beta;
        }

        // Step 4.0 --- UPDATE
        memmove(segment_lengths, segment_lengths + R, (8 - R) * sizeof(int));

        memmove(segment_buffer,
                segment_buffer + offset,
                (FFT_SIZE * 2 - offset) * sizeof(float));

        segments_ready -= R;
    }

    readpos += inNumSamples;
    memcpy(out, out_buffer + readpos, inNumSamples * sizeof(float));

    unit->P = P;
    unit->T = T;
    unit->pos = pos;
    unit->readpos = readpos;
    unit->segments_ready = segments_ready;
}

void PitchCorrection_Dtor(PitchCorrection * unit) {
    RTFree(unit->mWorld, unit->in_buffer);
    RTFree(unit->mWorld, unit->out_buffer);
    RTFree(unit->mWorld, unit->resampling_buffer);
    RTFree(unit->mWorld, unit->correlation_buffer);
    RTFree(unit->mWorld, unit->segment_buffer);
    RTFree(unit->mWorld, unit->tmp_buffer);
    RTFree(unit->mWorld, unit->freq_buffer);
    RTFree(unit->mWorld, unit->segment_lengths);
    RTFree(unit->mWorld, unit->pitch_marks);
}


PluginLoad(PitchCorrection) {
    ft = inTable;
    DefineDtorUnit(PitchCorrection);
}
