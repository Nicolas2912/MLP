// lcg.cpp
#include "lcg.h"

LCG::LCG(uint32_t seed) : a(1664525), c(1013904223), state(seed) {}

uint32_t LCG::next() {
    // The multiplication and addition will overflow naturally within uint32_t
    state = a * state + c;
    return state;
}

double LCG::random() {
    // Divide by 2^32 to get a value in [0,1)
    return static_cast<double>(next()) / static_cast<double>(4294967296);
}
