FIND_PATH(
	LIBSAMPLERATE_INCLUDE_DIR
	NAMES samplerate.h
)

FIND_LIBRARY(
	LIBSAMPLERATE_LIBRARY
	NAMES samplerate
)

SET(LIBSAMPLERATE_FOUND "NO")

IF( LIBSAMPLERATE_INCLUDE_DIR AND LIBSAMPLERATE_LIBRARY )
	SET(LIBSAMPLERATE_FOUND "YES")
ENDIF()