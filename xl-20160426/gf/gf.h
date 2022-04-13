#ifndef _GF_H
#define _GF_H

#include <stdint.h>

struct gf;

inline const gf _mul( const gf &a , const gf &b);

inline const gf _inv( const gf &a );

#define GF_FMT "%02X"

struct gf
{
   uint8_t v;

   static const unsigned p;

   inline gf() {}

   inline gf(const gf& a) : v(a.v) {}

   inline gf(const uint8_t &a) : v(a) {}

   inline const gf operator*(const gf &b) const 
   {
      return _mul(*this, b); 
   }

   inline const gf inv() const 
   { 
      return _inv(*this); 
   }

   inline operator uint8_t() const 
   {
      return v;
   }

   static inline const gf rand() 
   {
      return (::rand()&(p-1));
   }
};

#endif

