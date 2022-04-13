#ifndef XL_MEM_H__

#define XL_MEM_H__

#ifdef HAVE_HUGEPG

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000 /* arch specific */
#endif

void *XL_malloc(size_t size, void* addr = NULL) 
{
    size_t bytes = 
        ((size & ((1 << 30) - 1)) != 0) ? ((size & ~((1 << 30) - 1)) + (1 << 30)) : size;

    void *ret = mmap(addr, bytes, 
            PROT_READ|PROT_WRITE, MAP_HUGETLB|MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

    if (ret == MAP_FAILED)
    {
        fprintf(stderr, "\n\n");
        fprintf(stderr, "**************************************\n");
        fprintf(stderr, "*     Could not map huge pages!      *\n");
        fprintf(stderr, "**************************************\n\n");

        perror("mmap");

        ABORT;
    }

    ECHO("allocating %lu bytes (%lu Gb)\n", size, bytes >> 30);

    return ret;
}

template <class T>
void XL_free(T *p, size_t size)
{
    int err;

    size_t bytes = 
        ((size & ((1 << 30) - 1)) != 0) ? ((size & ~((1 << 30) - 1)) + (1 << 30)) : size;

    ECHO("freeing %lu bytes (%lu Gb)\n", size, bytes >> 30);

    err = munmap(p, bytes);

    if (err != 0)
        ECHO("Error during munmap!\n");
}

#else

#define XL_malloc(size) malloc(size)

#define XL_free(p, size) free(p)

#endif

#endif

