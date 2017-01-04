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

  ClearUnitOutputs(unit, 1);
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

PluginLoad(Autotune)
{
  ft = inTable;
  DefineSimpleUnit(Autotune);
}