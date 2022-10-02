#ifndef __MEMORY_OP_H__
#define __MEMORY_OP_H__

#include  <stddef.h>

#define   MEM_ALIGNED                 32

void*     dlcv_malloc_mem(size_t mem_size);
void      dlcv_free_mem(void* mem_ptr);

#endif