Autotune : UGen {
     *ar { arg scaleBuf, in, mul = 1.0, add = 0.0;
        ^this.multiNew('audio', scaleBuf, in).madd(mul, add);
     }
}