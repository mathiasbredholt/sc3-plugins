#include "SC_PlugIn.h"

static InterfaceTable *ft;

struct Autotune : public Unit
{
  float *buf;
};

extern "C"
{
  void Autotune_next(Autotune *unit, int inNumSamples);
  void Autotune_Ctor(Autotune *unit);
  void Autotune_Dtor(Autotune *unit);
}

void Autotune_Ctor(Autotune *unit)
{
  memset(unit->buf, 0, sizeof(float) * 5)

  SETCALC(Autotune_next);
  ClearUnitOutputs(unit, 1);
}

void Autotune_next(Autotune *unit, int inNumSamples)
{
  float *in = IN(0);
  float *out = OUT(0);
  float *buf = unit->buf;

  for (int i = 0; i < inNumSamples; ++i)
  {

    for (int j = 0; j < 5; j++)
    {
      if ((i - j) >= 0)
      {
        unit->buf[j] = in[i - j];
      }
    }

    out[i] = 0.2 * buf[0] + 0.2 * buf[1] + 0.2 * buf[2] + 0.2 * buf[3] + 0.2 * buf[4];
  }
}

void Autotune_Dtor(Autotune *unit)
{
  RTFree(unit->mWorld, unit->buf)
}

PluginLoad(Autotune)
{
  ft = inTable;
  DefineSimpleUnit(Autotune);
}