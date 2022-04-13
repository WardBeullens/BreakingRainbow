#ifndef GF2_H
#define GF2_H

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#include <cstdlib>
#include <cassert>
#include "gf.h"

const unsigned gf::p = 2;

inline const gf _mul( const gf &a , const gf &b)
{
   return a.v & b.v;   
}

inline const gf _inv( const gf  &a )
{
   assert( a.v != 0 );
   return a;
}

static const __m128i GFV2Mul[2] = {
   _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000),
   _mm_set_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
};

struct gfv_unit
{
   const static unsigned N = 128;
   static const unsigned B = 7;

   static const unsigned GF = 2;

   __m128i v;

   inline gfv_unit(){}

   inline gfv_unit(const __m128i& a) : v(a) {}

   inline gfv_unit& operator=(const gfv_unit &b) 
   { 
      v=b.v; 
      return *this;
   }

   inline void sum(const gfv_unit &a, const gfv_unit &b)
   {
      this->v = _mm_xor_si128(a.v, b.v);
   }

   inline gfv_unit& operator+=(const gfv_unit &b) 
   { 
      v = _mm_xor_si128(v, b.v); 
      return *this;
   }

   inline gfv_unit& operator-=(const gfv_unit &b) 
   { 
      v = _mm_xor_si128(v, b.v); 
      return *this;
   }

   inline gfv_unit operator-()
   {
      return *this;
   }

   inline void get(unsigned idx, uint8_t &v0, uint8_t &v1) const
   {
      uint8_t val = ((uint8_t*)&v)[(idx >> 3)];

      v0  = (uint8_t)(val >>  (0 + (idx & 6))) & 1;
      v1  = (uint8_t)(val >>  (1 + (idx & 6))) & 1;
   }

   inline void get(unsigned idx, uint8_t &v0, uint8_t &v1, 
         uint8_t &v2, uint8_t &v3) const
   {
      uint8_t val = ((uint8_t*)&v)[(idx >> 3)];

      v0  = (uint8_t)(val >>  (0 + (idx & 4))) & 1;
      v1  = (uint8_t)(val >>  (1 + (idx & 4))) & 1;
      v2  = (uint8_t)(val >>  (2 + (idx & 4))) & 1;
      v3  = (uint8_t)(val >>  (3 + (idx & 4))) & 1;
   }

   inline void get(unsigned idx, uint8_t &v0, uint8_t &v1, 
         uint8_t &v2, uint8_t &v3,
         uint8_t &v4, uint8_t &v5, uint8_t &v6, uint8_t &v7) const
   {
      uint8_t val = ((uint8_t*)&v)[(idx >> 3)];

      v0  = (uint8_t)(val >>  0) & 1;
      v1  = (uint8_t)(val >>  1) & 1;
      v2  = (uint8_t)(val >>  2) & 1;
      v3  = (uint8_t)(val >>  3) & 1;
      v4  = (uint8_t)(val >>  4) & 1;
      v5  = (uint8_t)(val >>  5) & 1;
      v6  = (uint8_t)(val >>  6) & 1;
      v7  = (uint8_t)(val >>  7) & 1;
   }

   inline void get(unsigned idx, uint8_t &v0, uint8_t &v1, 
         uint8_t &v2, uint8_t &v3,
         uint8_t &v4, uint8_t &v5, uint8_t &v6, uint8_t &v7,
         uint8_t &v8, uint8_t &v9, uint8_t &va, uint8_t &vb,
         uint8_t &vc, uint8_t &vd, uint8_t &ve, uint8_t &vf) const 
   {
      uint16_t val = (((uint16_t*)&v)[idx >> 4]);

      v0 = (uint8_t)(val >> 0x0) & 1;
      v1 = (uint8_t)(val >> 0x1) & 1;
      v2 = (uint8_t)(val >> 0x2) & 1;
      v3 = (uint8_t)(val >> 0x3) & 1;
      v4 = (uint8_t)(val >> 0x4) & 1;
      v5 = (uint8_t)(val >> 0x5) & 1;
      v6 = (uint8_t)(val >> 0x6) & 1;
      v7 = (uint8_t)(val >> 0x7) & 1;
          
      v8 = (uint8_t)(val >> 0x8) & 1;
      v9 = (uint8_t)(val >> 0x9) & 1;
      va = (uint8_t)(val >> 0xa) & 1;
      vb = (uint8_t)(val >> 0xb) & 1;
      vc = (uint8_t)(val >> 0xc) & 1;
      vd = (uint8_t)(val >> 0xd) & 1;
      ve = (uint8_t)(val >> 0xe) & 1;
      vf = (uint8_t)(val >> 0xf) & 1;

        //get(idx + 0, v0, v1, v2, v3, v4, v5, v6, v7);
      //get(idx + 8, v8, v9, va, vb, vc, vd, ve, vf);
    }

   inline bool is_zero() const 
   { 
      __m128i a = _mm_or_si128(v,_mm_srli_si128(v,8)); 
      return 0 == _mm_cvtsi128_si32(_mm_or_si128(a,_mm_srli_si128(a,4))); 
   }   
   
   inline static const gfv_unit rand() 
   { 
      return _mm_set_epi16(::rand(), ::rand(), ::rand(), ::rand(), 
                                     ::rand(), ::rand(), ::rand(), ::rand());  
   }

   inline gfv_unit& operator*=(const gf &b) 
   { 
      this->v = _mm_and_si128(this->v, GFV2Mul[b.v]);

      return *this;
   }

   inline void set(unsigned i, const gf &a) 
   {
#ifdef __SSSE3__ 
       char b = _mm_cvtsi128_si64(_mm_shuffle_epi8(v, _mm_cvtsi32_si128(i >> 3)));
       ((char *)&v)[i >> 3] = b ^ ((b ^ (a.v << (i & 7))) & (1 << (i & 7)));
#else
       if(a.v)  ((char *)&v)[i >> 3] |= (1 << (i & 7));
       else     ((char *)&v)[i >> 3] &= ~(1 << (i & 7));
#endif
   }

   inline const gf operator[](unsigned idx) const 
   {
       uint8_t* tmp = (uint8_t*)(&this->v);

       uint8_t v0 = tmp[(idx >> 3)];
       v0 >>= idx & 7;
       v0 &= 1;

       return v0;
   }

};

#endif
