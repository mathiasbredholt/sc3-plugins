#include "SC_PlugIn.h"

static InterfaceTable *ft;

struct Autotune : public Unit
{
};

extern "C"
{
  void Autotune_Ctor(Autotune *unit);
  void Autotune_next(Autotune *unit, int inNumSamples);
}

void Autotune_Ctor( Autotune *unit )
{
  SETCALC(Autotune_next);

  ZOUT0(0) = 0;
}

void Autotune_next( Autotune *unit, int inNumSamples )
{
  float *in = IN(0);
  float *out = OUT(0);

  float val;

  for ( int i = 0; i < inNumSamples; ++i) {
    val = in[i];
    out[i] = val;
  }
}

PluginLoad(M2)
{
  ft = inTable;
  DefineSimpleUnit(Autotune);
}