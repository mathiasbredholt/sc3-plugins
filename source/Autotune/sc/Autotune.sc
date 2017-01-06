Autotune : UGen {
	 *ar { arg bufnum, in, mul = 1.0, add = 0.0;
	 ^this.multiNew('audio', bufnum, in).madd(mul, add)
	 }
}