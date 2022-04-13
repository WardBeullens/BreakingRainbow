#ifndef _GFV_H
#define _GFV_H

#include <stdio.h>

#include "gf_buf.h"

template <unsigned n>
struct gfv
{
   typedef gfv_unit unit;

   static const unsigned len = n;

   static const unsigned M = (n+unit::N-1)/unit::N;

   static const unsigned GF = unit::GF;

   unit c[M];

   inline bool is_zero() const 
   { 
      if (M > 1)
      {
         unit u; 

         u.v = c[M-2].v; 

         for (unsigned i = M-2; i--; ) 
            u.v = _mm_or_si128(u.v, c[i].v);

         if (!u.is_zero())
            return false;

         uint8_t ret = 0;

         for (unsigned t = 0; t < n - (M-1)*unit::N; t++)
            ret |= c[M-1][t];

         return (ret == 0);
      }
      else
      {
         uint8_t ret = 0;

         for (unsigned t = 0; t < n; t++)
            ret |= c[0][t];

         return (ret == 0);
      }
   }

   inline gfv<n>(){}

   inline gfv<n>(const gfv<n>& a)
   { 
      for (unsigned i = M; i--; ) 
         c[i] = a.c[i]; 
   }

   inline gfv<n>(const gf& a) 
   {
      for (unsigned i = M; i--; ) 
         c[i] = a; 
   }

   inline gfv<n>& operator=(const gfv<n> &b) 
   { 
      for (unsigned i = M; i--; )
         c[i] = b.c[i]; 
      return *this;
   }

   inline bool operator==(const gfv<n> &b)
   {
      for (unsigned i = M; i--; )
         if (c[i] != b.c[i])
            return false;

      return true;
   }

   inline bool operator!=(const gfv<n> &b)
   {
      return !(*this == b);
   }

   inline gfv<n>& operator+=(const gfv<n> &b)
   {
      for (unsigned i = M; i--; ) 
         c[i] += b.c[i]; 

      return *this;
   }

   inline void add_nored(const gfv<n> &b)
   {
      for (unsigned i = M; i--; ) 
         c[i].add_nored(b.c[i]);
   }

   inline void reduce()
   {
      for (unsigned i = M; i--; ) 
         c[i].reduce();
   }

   inline void part_reduce()
   {
      for (unsigned i = M; i--; ) 
         c[i].part_reduce();
   }

   inline gfv<n>& operator-=(const gfv<n> &b)
   {
      for (unsigned i = M; i--; ) 
         c[i] -= b.c[i]; 

      return *this;
   }

   inline gfv<n> operator-()
   {
      gfv<n> ret;

      for (unsigned i = M; i--; ) 
         ret.c[i] = - this->c[i]; 

      return ret;
   }

   inline const gfv<n> operator+(const gfv<n> &b) const 
   {
      gfv<n> r = (*this); 
      r += b; 
      return r;
   }

   inline gfv<n>& operator*=(const gf &b) 
   {
      for (unsigned i = M; i--; ) 
         c[i] *= b; 
      return *this; 
   }

   inline gfv<n>& operator*=(const gfv<n>&b) 
   { 
      for (unsigned i = M; i--; ) 
         c[i] *= b.c[i];
      return *this; 
   }

   inline const gfv<n> operator*(const gf &b) const 
   { 
      gfv<n> r=(*this); 
      r*=b; 
      return r; 
   }

   inline void mad(const gfv<n> &a, 
         const gf &b) 
   {
      for(unsigned i = 0; i < M; i++)
         c[i].mad(a.c[i], b);
   }

   inline void set_zero() 
   { 
      for (unsigned i = 0; i < M; i++) 
         c[i]=_mm_setzero_si128(); 
   }

   inline void set( unsigned i, const gf & a )
   { 
      c[i >> unit::B].set(i & (unit::N-1), a); 
   }

   inline void set( unsigned i, uint8_t v0, uint8_t v1)
   {
      c[i >> unit::B].set(i & (unit::N-1), v0, v1); 
   }

   inline void set( unsigned i, uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3)
   { 
      c[i >> unit::B].set(i & (unit::N-1), v0, v1, v2, v3); 
   }

   inline const gf operator[](uint64_t i) const 
   {
      return (c[i >> unit::B])[i & (unit::N-1)]; 
   }

   inline gf get(unsigned i) const
   {
      return (c[i >> unit::B])[i & (unit::N-1)];
   }

   inline void get(unsigned i, uint8_t &v0, uint8_t &v1) const 
   {
      c[i >> unit::B].get(i & (unit::N-1), v0, v1); 
   }

   inline void get(unsigned i, uint8_t &v0, uint8_t &v1, 
         uint8_t &v2, uint8_t &v3) const 
   {
      c[i >> unit::B].get(i & (unit::N-1), v0, v1, v2, v3); 
   }

   inline void get(unsigned i, uint8_t &v0, uint8_t &v1, 
         uint8_t &v2, uint8_t &v3,
         uint8_t &v4, uint8_t &v5, uint8_t &v6, uint8_t &v7) const 
   {
      c[i >> unit::B].get(i & (unit::N-1), v0, v1, v2, v3, v4, v5, v6, v7); 
   }

   inline void get(unsigned i, uint8_t &v0, uint8_t &v1, 
         uint8_t &v2, uint8_t &v3,
         uint8_t &v4, uint8_t &v5, uint8_t &v6, uint8_t &v7,
         uint8_t &v8, uint8_t &v9, uint8_t &v10, uint8_t &v11,
         uint8_t &v12, uint8_t &v13, uint8_t &v14, uint8_t &v15) const 
   {
      c[i >> unit::B].get(i & (unit::N-1), v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15); 
   }

   inline const gf dot(const gfv<n> &b) const 
   { 
      gfv<n> rv=(*this)*b;

      if( n&15 ) 
         rv.c[M-1].v = _mm_slli_si128(rv.c[M-1].v,16-(n&15));

      for(unsigned i=1; i<M; i++) 
         rv.c[0] += rv.c[i];

      return rv.c[0].add_up(); 
   }

   void dump(FILE * fp) const 
   { 
      for(unsigned i=0;i<n;i++) 
         fprintf(fp, GF_FMT " ",(int)((*this)[i].v)); 

      fprintf(fp,"\n"); 
   }

   inline void rand() 
   {
      for(unsigned i=M; i--;) 
         c[i] = unit::rand(); 
   }

   inline void rand(uint32_t row_weight)
   {
      this->set_zero();

      for (unsigned i = 0; i < row_weight; i++)
      {
         unsigned idx = 0;

         do
         {
            idx = ::rand() % n;
         } while (this->get(idx).v != 0);

         this->set(idx, 1);
      }
   }

   inline const gf hash()
   {
      gf ret;

      ret.v = 0;
      for(unsigned i = 0; i < n; i++)
         ret += (*this)[i];

      return ret;
   }

   unsigned weight()
   {
      unsigned ret = 0;

      for(unsigned i = 0; i < n; i++)
         if( (*this)[i].v )
            ret++;

      return ret;
   }

   inline gfv<n> &prod2(const gfv<n> &v)
   {
      for (unsigned i = M; i--; )
         c[i].prod2(v.c[i]);
      return *this;
   }

   inline gfv<n> &prod4(const gfv<n> &v)
   {
      for (unsigned i = M; i--; )
         c[i].prod4(v.c[i]);
      return *this;
   }

   inline gfv<n> &prod8(const gfv<n> &v)
   {
      for (unsigned i = M; i--; )
         c[i].prod8(v.c[i]);

      return *this;
   }


   inline void prod(const gfv<n> &v, const gf &b)
   {
      for (unsigned i = M; i--; )
         c[i].prod(v.c[i], b);
   }

   inline void sum(const gfv<n> &v, const gfv<n> &w)
   { 
      for (unsigned i = M; i--; )
         c[i].sum(v.c[i], w.c[i]);
   }

#if QQ == 31
   inline void reduce(gfv<n> * buf, bool add)
   {
      for (unsigned i = 30; i > 3; i -= 3)
      {
         buf[i-1].add_nored(buf[i]); // 60
         buf[1].add_nored(buf[i]);   // 90

         buf[i-2].add_nored(buf[i-1]); // 90
         buf[1].add_nored(buf[i-1]);   // 150

         buf[i-3].add_nored(buf[i-2]);  // 120
         buf[i-3].part_reduce();
         buf[1].add_nored( buf[i-2]);  // 270
         buf[1].part_reduce();
      }

      buf[2].add_nored(buf[3]);
      buf[1].add_nored(buf[3]);

      buf[1].add_nored(buf[2]);
      buf[1].add_nored(buf[2]);

      buf[1].reduce();

      if (add)
         *this += buf[1];
      else
         *this = buf[1];
   }
#endif

#if QQ == 16

   // should be parallelized?
   inline void gf16_expand(gfv<n> * table) const
   {
      table[1] = (*this);

#ifdef __SSSE3__
      table[2].prod((*this), gf(2));
      table[4].prod((*this), gf(4));
      table[8].prod((*this), gf(8));
#else
      table[2].prod2((*this));
      table[4].prod4((*this));
      table[8].prod8((*this));
#endif

      table[3].sum(table[2], table[1]);

      table[6].sum(table[2], table[4]);
      table[7].sum(table[6], table[1]);
      table[5].sum(table[7], table[2]);

      table[12].sum(table[4], table[8]);
      table[13].sum(table[12], table[1]);
      table[15].sum(table[13], table[2]);
      table[14].sum(table[15], table[1]);

      table[10].sum(table[14], table[4]);
      table[11].sum(table[10], table[1]);
      table[9].sum(table[11], table[2]);
   }

   // should be parallelized?
   inline void reduce(gfv<n> * table, bool add)
   {
      table[8] += table[15];
      table[7] += table[15];
      table[8] += table[14];
      table[6] += table[14];
      table[8] += table[13];
      table[5] += table[13];
      table[8] += table[12];
      table[4] += table[12];
      table[8] += table[11];
      table[3] += table[11];
      table[8] += table[10];
      table[2] += table[10];
      table[8] += table[ 9];
      table[1] += table[ 9];

      table[4] += table[ 7];
      table[3] += table[ 7];
      table[4] += table[ 6];
      table[2] += table[ 6];
      table[4] += table[ 5];
      table[1] += table[ 5];

      table[2] += table[ 3];
      table[1] += table[ 3];

      if (add)
         (*this) += table[1];
      else
         (*this) = table[1];

#ifdef __SSSE3__
      (*this) += table[2] * gf(2); // why is * faster than *= ?
      (*this) += table[4] * gf(4); 
      (*this) += table[8] * gf(8);
#else
      (*this) += table[2].prod2(table[2]);
      (*this) += table[4].prod4(table[4]);
      (*this) += table[8].prod8(table[8]);
#endif
   }

#endif // QQ == 16

};

#endif // _GFV_H

