#ifndef __ALIGNMALLOC_H__
#define __ALIGNMALLOC_H__

#define MALLOC_ALIGN 16

template<typename _Tp>
static inline _Tp *alignPtr(_Tp *ptr, int n = (int) sizeof(_Tp)) {
	return (_Tp *)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n) {
	return (sz + n - 1) & -n;
}

static inline void *fastMalloc(size_t size) {
	unsigned char *udata = (unsigned char *)malloc(size + sizeof(void *) + MALLOC_ALIGN);
	if (!udata)
		return 0;
	unsigned char **adata = alignPtr((unsigned char **)udata + 1, MALLOC_ALIGN);
	adata[-1] = udata;
	return adata;
}

static inline void fastFree(void *ptr) {
	if (ptr) {
		unsigned char *udata = ((unsigned char **)ptr)[-1];
		free(udata);
	}
}

#endif