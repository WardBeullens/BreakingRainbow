#ifndef __GF_BUF_H__
#define __GF_BUF_H__

#if QQ == 31
template <class gfv>
class gf_buf_line
{
   public:

      gfv vec;
      uint8_t cnt;

      inline gf_buf_line<gfv>& operator+=(const gfv &a)
      {
         if (cnt > (255-30))
         {
            vec.part_reduce();
            cnt = 38;
         }

         vec.add_nored(a);
         cnt += 30;

         return *this;
      }

      inline gf_buf_line<gfv>& operator+=(gf_buf_line<gfv> &a)
      {
         if ((a.cnt+cnt) > (255-30))
         {
            a.vec.part_reduce();
            a.cnt = 38;
         }
         if ((a.cnt+cnt) > (255-30))
         {
            vec.part_reduce();
            cnt = 38;
         }

         vec.add_nored(a.vec);
         cnt += a.cnt;

         return *this;
      }

      inline operator gfv()
      {
         if (cnt > 30)
         {
            vec.reduce();
            cnt = 30;
         }

         return vec;
      }

      inline void set_zero()
      {
         vec.set_zero();
      }
};

template <class gfv>
class gf_buf
{
   public:
        gf_buf_line<gfv> buf[gfv::GF];

        inline gf_buf_line<gfv>& operator[](unsigned i)
        {
           return buf[i];
        }

        inline void reduce(gfv &a, bool add)
        {
            for (unsigned i = 30; i > 1; i--)
            {
               buf[i-1] += buf[i];
               buf[1] += buf[i];
            }

            if (add)
               a += buf[1];
            else
               a = buf[1];
        }

};

#else // #if QQ == 31

template <class gfv>
class gf_buf
{
   public:
        gfv buf[gfv::GF];

        inline gfv& operator[](unsigned i)
        {
           return buf[i];
        }

        inline void reduce(gfv &a, bool add)
        {
            a.reduce(buf, add);
        }

};
#endif // #if QQ == 31

#endif //ifndef __GF_BUF_H__

