Autotune : UGen {
	 *ar { arg bufnum, in, freq = 440, mul = 1.0, add = 0.0;
	 ^this.multiNew('audio', bufnum, in, freq).madd(mul, add)
	 }
}