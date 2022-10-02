#include "memory_op.h"
#include <stdlib.h>

#ifndef ALLOC_MEM_MAX
#define ALLOC_MEM_MAX  0x7FFFFFFF
#endif

void*  dlcv_malloc_mem(size_t mem_size)
{
	void*           ptr  = 0 ;
    unsigned char   diff = 0 ;

	if (mem_size > (ALLOC_MEM_MAX - MEM_ALIGNED) || mem_size <= 0)
		return NULL;

	ptr = malloc(mem_size + MEM_ALIGNED) ;
	if (!ptr)
		return ptr;

	diff = (unsigned char)(((~(unsigned long long)ptr)&((unsigned long long)(MEM_ALIGNED - 1))) + 1) ;

	ptr  = (unsigned char *)ptr + diff ;
	((unsigned char *)ptr)[-1] = (unsigned char)diff ;

	return ptr;
}

void   dlcv_free_mem(void* mem_ptr)
{
	if (mem_ptr)
	{
		int v = ((unsigned char *)mem_ptr)[-1];
		if (v <= 0 || v > MEM_ALIGNED)   return;
		//assert(v > 0 && v <= MEM_ALIGNED) ;
		free((unsigned char *)mem_ptr - v);
	}
}