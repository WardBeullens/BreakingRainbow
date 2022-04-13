#ifndef _DEFINE_H_
#define _DEFINE_H_

#ifdef OPEN_MPI
  #if QQ == 2
    static const unsigned BW_M = 512 * MPI_BLOCKS;
  #else // #if QQ == 2
    static const unsigned BW_M = 128 * MPI_BLOCKS;
  #endif // #if QQ == 2
#else // #ifdef OPEN_MPI
  #if QQ == 2
    static const unsigned BW_M = 512;
  #else // #if QQ == 2
    static const unsigned BW_M = 128;
  #endif // #if QQ == 2
#endif // #ifdef OPEN_MPI

   static const unsigned BW_N = BW_M;
   static const unsigned NSOL = 1;

#if QQ == 2
   static const unsigned X_ROW_WEIGHT = 15;
   static const unsigned Z_ROW_WEIGHT = 15;
#else // #if QQ == 2
   static const unsigned X_ROW_WEIGHT = 16;
   static const unsigned Z_ROW_WEIGHT = 16;
#endif // #if QQ == 2

#endif

