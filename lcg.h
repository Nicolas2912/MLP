// lcg.h
#ifndef LCG_H
#define LCG_H

#include <cstdint>

class LCG {
public:
    LCG(uint32_t seed = 42);
    uint32_t next();
    double random(); // Returns a double in [0, 1)

private:
    uint32_t a;
    uint32_t c;
    uint32_t state;
};

#endif // LCG_H
