#ifndef _GF32_H
#define _GF32_H

#include <stdio.h>

#include <emmintrin.h>

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#include "gf.h"

const unsigned gf::p = 31;

#undef GF_FMT

#define GF_FMT "%02d"

const uint8_t gf31_inv_tab[31] = 
   {  0,  1, 16, 21,  8, 25, 26,  9, 
      4,  7, 28, 17, 13, 12, 20, 29,  
      2, 11, 19, 18, 14,  3, 24, 27, 
      22, 5,  6, 23, 10, 15, 30};

inline const gf _mul( const gf &a, const gf &b)
{
   return ((uint32_t)a.v * (uint32_t)b.v) % 31;
}

inline const gf _inv(const gf &a)
{ 
   if ((a.v % 31) == 0) 
   { 
      puts("inv of 0"); 
      exit(-1); 
   }

   return gf31_inv_tab[a.v % 31];
}


struct gfv_unit
{
   const static unsigned N = 16;
   static const unsigned B = 4;

   static const unsigned GF = 31;

   __m128i v;

   inline gfv_unit(){}

   inline gfv_unit(const __m128i& a) : v(a) {}

   inline gfv_unit& operator=(const gfv_unit &b) 
   { 
      v=b.v; 
      return *this;
   }

   void dump()
   {
      for (int i = 0; i < 16; i++)
      {
         uint8_t v = ((uint8_t*)&(this->v))[i];

         printf("%i, ", v);
      }

      printf("\n");
   }

   inline static void reduce16_8(__m128i &v)
   {
      __m128i mask  = _mm_set1_epi16(0x001F);
      __m128i mask2 = _mm_set1_epi16(0x03E0);

      __m128i low = _mm_and_si128(mask, v);

      __m128i high = _mm_and_si128(v, mask2);
      high = _mm_srli_epi16(high, 5);

      v = _mm_add_epi16(high, low);
   }

   inline static void reduce7(__m128i &v)
   {
      __m128i mask  = _mm_set1_epi16(0x001F);
      __m128i mask2 = _mm_set1_epi16(0x03E0);

      __m128i low = _mm_and_si128(mask, v);

      __m128i high = _mm_and_si128(v, mask2);
      high = _mm_srli_epi16(high, 5);

      v = _mm_add_epi16(high, low);

      __m128i cmp = _mm_cmpeq_epi16(mask, v);

      v = _mm_andnot_si128(cmp, v);
   }

   inline static void reduce16(__m128i &v)
   {
//      __m128i mask  = _mm_set1_epi16(0x001F);
//      __m128i mask2 = _mm_set1_epi16(0x03E0);
//
//      __m128i low = _mm_and_si128(mask, v);
//
//      __m128i high = _mm_and_si128(v, mask2);
//      high = _mm_srli_epi16(high, 5);
//
//      v = _mm_add_epi16(high, low);
//
//      low = _mm_and_si128(mask, v);
//
//      high = _mm_and_si128(v, mask2);
//      high = _mm_srli_epi16(high,5);
//
//      v = _mm_add_epi16(high, low);
//
//      __m128i cmp = _mm_cmpeq_epi16(mask, v);
//
//      v = _mm_andnot_si128(cmp, v);

      __m128i p31 = _mm_set1_epi16(31);
      __m128i pinv = _mm_set1_epi16(65536/31);

      __m128i tmp = _mm_mulhi_epi16(v, pinv);
      tmp = _mm_mullo_epi16(tmp, p31);
      v = _mm_sub_epi16(v, tmp);
   }

   inline static void reduce(__m128i &v)
   {
      __m128i mask = _mm_set1_epi8(0x1F);
      __m128i mask2 = _mm_set1_epi8(0xE0);

      __m128i low = _mm_and_si128(mask, v);

      v = _mm_and_si128(v, mask2);
      v = _mm_srli_epi16(v,5);

      v = _mm_add_epi8(v, low);

      low = _mm_and_si128(mask, v);

      v = _mm_and_si128(v, mask2);
      v = _mm_srli_epi16(v,5);

      v = _mm_add_epi8(v, low);

      __m128i cmp = _mm_cmpeq_epi8(mask, v);

      v = _mm_andnot_si128(cmp, v);
   }

   union test
   {
      __m128i i128;
      uint32_t i32[4];
      uint16_t i16[8];
      uint8_t i8[16];
   };

   inline void mad(__m128i &b0, __m128i &b1, __m128i &b2, __m128i &b3, 
         const gf &_b) const
   {
      __m128i b = _mm_set1_epi16(_b);

      __m128i mask = _mm_set1_epi16(0x00FF);

      __m128i a02 = _mm_and_si128(v, mask);
      __m128i a13 = _mm_srli_epi16(v,8);

      __m128i p02 = _mm_mullo_epi16(a02, b);
      __m128i p13 = _mm_mullo_epi16(a13, b);

      mask = _mm_set1_epi32(0xFFFF);

      __m128i tmp;

      tmp = _mm_and_si128(p02, mask);
      b0 = _mm_add_epi32(b0, tmp);

      tmp = _mm_srli_epi32(p02,16);
      b2 = _mm_add_epi32(b2, tmp);

      tmp = _mm_and_si128(p13, mask);
      b1 = _mm_add_epi32(b1, tmp);

      tmp = _mm_srli_epi32(p13,16);
      b3 = _mm_add_epi32(b3, tmp);
   }

   inline void reduce(__m128i b0, __m128i b1, __m128i b2, __m128i b3)
   {
      __m128i m30 = _mm_set1_epi32(0x3FFFFFE0);
      __m128i m26 = _mm_set1_epi32(0x03FFFFE0);
      __m128i m22 = _mm_set1_epi32(0x003FFFE0);
      __m128i m18 = _mm_set1_epi32(0x0003FFE0);
      __m128i m14 = _mm_set1_epi32(0x00003FE0);
      __m128i m5  = _mm_set1_epi32(0x0000001F);

      __m128i h = _mm_and_si128(b0, m30);
      h = _mm_srli_epi32(h, 5);
      __m128i l = _mm_and_si128(b0, m5);
      b0 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b0, m26);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b0, m5);
      b0 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b0, m22);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b0, m5);
      b0 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b0, m18);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b0, m5);
      b0 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b0, m14);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b0, m5);
      b0 = _mm_add_epi32(h, l);

      reduce16(b0);


      h = _mm_and_si128(b2, m30);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b2, m5);
      b2 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b2, m26);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b2, m5);
      b2 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b2, m22);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b2, m5);
      b2 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b2, m18);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b2, m5);
      b2 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b2, m14);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b2, m5);
      b2 = _mm_add_epi32(h, l);

      reduce16(b2);

      b2 = _mm_slli_epi32(b2, 16);

      __m128i b02 = _mm_or_si128(b0, b2);


      h = _mm_and_si128(b1, m30);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b1, m5);
      b1 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b1, m26);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b1, m5);
      b1 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b1, m22);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b1, m5);
      b1 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b1, m18);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b1, m5);
      b1 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b1, m14);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b1, m5);
      b1 = _mm_add_epi32(h, l);

      reduce16(b1);


      h = _mm_and_si128(b3, m30);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b3, m5);
      b3 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b3, m26);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b3, m5);
      b3 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b3, m22);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b3, m5);
      b3 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b3, m18);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b3, m5);
      b3 = _mm_add_epi32(h, l);

      h = _mm_and_si128(b3, m14);
      h = _mm_srli_epi32(h, 5);
      l = _mm_and_si128(b3, m5);
      b3 = _mm_add_epi32(h, l);

      reduce16(b3);

      b3 = _mm_slli_epi32(b3, 16);

      __m128i b13 = _mm_or_si128(b1, b3);


      b13 = _mm_slli_epi16(b13, 8);

      v = _mm_or_si128(b13, b02);

   }

   inline void mad(const gfv_unit &a, const gf &_b)
   {
#ifdef __SSSE3__
      __m128i b = _mm_set1_epi16(_b + 256);


      __m128i al = _mm_and_si128(a.v, _mm_set1_epi16(0x00FF));
      __m128i vl = _mm_and_si128(this->v, _mm_set1_epi16(0x00FF));
      vl = _mm_slli_epi16(vl, 8);

      vl = _mm_or_si128(vl, al);

      __m128i m1 = _mm_maddubs_epi16(vl, b);
      reduce16(m1);



      __m128i ah = _mm_and_si128(a.v, _mm_set1_epi16(0xFF00));
      __m128i vh = _mm_and_si128(this->v, _mm_set1_epi16(0xFF00));
      ah = _mm_srli_epi16(ah, 8);

      vh = _mm_or_si128(vh, ah);

      __m128i m2 = _mm_maddubs_epi16(vh, b);
      reduce16(m2);

      m2 = _mm_slli_epi16(m2, 8);


      this->v = _mm_or_si128(m1, m2);
#else
      __m128i b = _mm_set1_epi16(_b);

      __m128i al = _mm_and_si128(a.v, _mask_low);
      __m128i cl = _mm_and_si128(this->v, _mask_low);

      __m128i m1 = _mm_mullo_epi16(al, b);
      m1 = _mm_add_epi16(m1, cl);

      reduce16(m1);

      __m128i ah = _mm_and_si128(a.v, _mask_high);
      __m128i ch = _mm_and_si128(this->v, _mask_high);
      ah = _mm_srli_epi16(ah, 8);
      ch = _mm_srli_epi16(ch, 8);

      __m128i m2 = _mm_mullo_epi16(ah, b);
      m2 = _mm_add_epi16(m2, ch);

      reduce16(m2);

      m2 = _mm_slli_epi16(m2, 8);
      
      this->v = _mm_or_si128(m1, m2);
#endif
   }


   inline void prod(const gfv_unit &a, const gf &_b)
   {
//      uint16_t res[16];//((uint8_t*)&(this->v));
//      uint8_t* ap = ((uint8_t*)&(a.v));
//
//      for (unsigned i = 0; i < 16; i++)
//         res[i] = ((uint32_t)ap[i] * (uint32_t)_b) % 31;

#ifdef __SSSE3__
      __m128i b = _mm_set1_epi16(_b);

      __m128i m1 = _mm_maddubs_epi16(a.v, b);
      reduce16(m1);

      b = _mm_set1_epi16(_b << 8);

      __m128i m2 = _mm_maddubs_epi16(a.v, b);
      reduce16(m2);

      m2 = _mm_slli_epi16(m2, 8);

      this->v = _mm_or_si128(m1, m2);
#else
      __m128i b = _mm_set1_epi16(_b);

      __m128i al = _mm_and_si128(a.v, _mask_low);
      __m128i m1 = _mm_mullo_epi16(al, b);

      reduce16(m1);

      __m128i ah = _mm_and_si128(a.v, _mask_high);
      ah = _mm_srli_epi16(ah, 8);
      __m128i m2 = _mm_mullo_epi16(ah, b);

      reduce16(m2);

      m2 = _mm_slli_epi16(m2, 8);
      
      this->v = _mm_or_si128(m1, m2);
#endif
   }

   inline void sum(const gfv_unit &a, const gfv_unit &b)
   {
      __m128i mask = _mm_set1_epi8(0x1F);
      __m128i mask2 = _mm_set1_epi8(0xE0);

      this->v = _mm_add_epi8(a.v, b.v);

      // reduce start - small reduce assuming all inputs are reduced
      __m128i low = _mm_and_si128(this->v, mask);

      this->v = _mm_and_si128(this->v, mask2);
      this->v = _mm_srli_epi16(this->v,5);

      this->v = _mm_add_epi8(this->v, low);

      __m128i cmp = _mm_cmpeq_epi8(this->v, mask);

      this->v = _mm_andnot_si128(cmp, this->v);
      // reduce end
   }

   inline gfv_unit& operator+=(const gfv_unit &b) 
   { 
      sum(this->v, b);

      return *this;
   }

   inline void reduce()
   {
      reduce(this->v);
   }

   inline void part_reduce()
   {
      __m128i mask = _mm_set1_epi8(0x1F);
      __m128i mask2 = _mm_set1_epi8(0xE0);

      __m128i low = _mm_and_si128(mask, this->v);

      this->v = _mm_and_si128(this->v, mask2);
      this->v = _mm_srli_epi16(this->v,5);

      this->v = _mm_add_epi8(this->v, low);

      __m128i cmp = _mm_cmpeq_epi8(mask, this->v);

      this->v = _mm_andnot_si128(cmp, this->v);
   }

   inline void add_nored(const gfv_unit &b)
   {
      this->v = _mm_add_epi8(this->v, b.v);
   }

   inline gfv_unit& operator-=(const gfv_unit &b) 
   { 
      uint8_t* res = ((uint8_t*)&(this->v));
      uint8_t* bp = ((uint8_t*)&(b.v));

      for (unsigned i = 0; i < 16; i++)
         res[i] = (31 + res[i] - bp[i]) % 31;

      return *this;
   }

   inline gfv_unit operator-()
   {
      gfv_unit ret;

      for (unsigned i = 0; i < 16; i++)
         ((uint8_t*)&(ret.v))[i] = (31 - ((uint8_t*)&(this->v))[i]) % 31;

      return ret;
   }

   inline void get(unsigned idx, uint8_t &v0, uint8_t &v1, 
         uint8_t &v2, uint8_t &v3,
         uint8_t &v4, uint8_t &v5, uint8_t &v6, uint8_t &v7,
         uint8_t &v8, uint8_t &v9, uint8_t &va, uint8_t &vb,
         uint8_t &vc, uint8_t &vd, uint8_t &ve, uint8_t &vf) const 
   {
      uint8_t* val = ((uint8_t*)&(this->v));

      v0 = val[0x0];
      v1 = val[0x1];
      v2 = val[0x2];
      v3 = val[0x3];
      v4 = val[0x4];
      v5 = val[0x5];
      v6 = val[0x6];
      v7 = val[0x7];
      v8 = val[0x8];
      v9 = val[0x9];
      va = val[0xa];
      vb = val[0xb];
      vc = val[0xc];
      vd = val[0xd];
      ve = val[0xe];
      vf = val[0xf];
   }

   inline bool is_zero() const 
   { 
      __m128i a = _mm_or_si128(v,_mm_srli_si128(v,8)); 
      return 0 == _mm_cvtsi128_si32(_mm_or_si128(a,_mm_srli_si128(a,4))); 
//      for (unsigned i = 0; i < 16; i++)
//         if ((((uint8_t*)&v)[i] > 0) && (((uint8_t*)&v)[i] < 31))
//            return false;
//
//      return true;
   }   
   
   inline static const gfv_unit rand() 
   { 
      return _mm_set_epi8(::rand()%31, ::rand()%31, ::rand()%31, ::rand()%31,
                                     ::rand()%31, ::rand()%31, ::rand()%31, ::rand()%31,
            ::rand()%31, ::rand()%31, ::rand()%31, ::rand()%31,
                                     ::rand()%31, ::rand()%31, ::rand()%31, ::rand()%31);  
   }

   inline gfv_unit& operator*=(const gf &b) 
   { 
      this->prod(*this, b);

      return *this;
   }

   inline void set(unsigned i, const gf &a) 
   { 
      ((uint8_t*)&v)[i] = a.v % 31;
   }

   inline const gf operator[](unsigned i) const 
   {
      return ((uint8_t*)&v)[i] % 31;
   }
};

#endif

