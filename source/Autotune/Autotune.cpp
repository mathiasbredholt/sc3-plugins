#include "SC_PlugIn.h"

static InterfaceTable *ft;

struct Autotune : public Unit
{
  // float *buf;
};

extern "C"
{
  void Autotune_next(Autotune *unit, int inNumSamples);
  void Autotune_Ctor(Autotune *unit);
  void Autotune_Dtor(Autotune *unit);
}

void Autotune_Ctor(Autotune *unit)
{
  // unit->buf = (float*) RTAlloc(unit->mWorld, 20 * sizeof(float));
  // memset(unit->buf, 0, 20 * sizeof(float));

  SETCALC(Autotune_next);
  ClearUnitOutputs(unit, 1);
}

void Autotune_next(Autotune *unit, int inNumSamples)
{
  float *in = IN(0);
  float *out = OUT(0);
  // float *buf = unit->buf;

  float tmp;

  for (int i = 0; i < inNumSamples; ++i)
  {
    tmp = 0;

    for (int j = 0; j < 30; j++)
    {
      if ((i - j) >= 0)
      {
        // unit->buf[j] = in[i - j];
        tmp += 1/30 * in[i - j];
      }
    }

    out[i] = tmp;

    // out[i] = 0.2 * buf[0] + 0.2 * buf[1] + 0.2 * buf[2] + 0.2 * buf[3] + 0.2 * buf[4];
  }
}

void Autotune_Dtor(Autotune *unit)
{
  // RTFree(unit->mWorld, unit->buf);
}

PluginLoad(Autotune)
{
  ft = inTable;
  DefineDtorUnit(Autotune);
}