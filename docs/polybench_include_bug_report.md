# Bug Report: `papi_counters.list` Include Path Resolution Shadowing

## Description
When compiling PolyBench/C with PAPI support (`-DPOLYBENCH_PAPI`), the inclusion of `papi_counters.list` in `utilities/polybench.c` uses double quotes instead of angle brackets. This causes the compiler to resolve the include file relative to the source file's directory (`utilities/`) before checking any user-provided `-I` include directories.

Because PolyBench distributes a default `utilities/papi_counters.list` file, it becomes impossible for a user to dynamically generate their own `papi_counters.list` in a separate build directory and include it via `-I <path>`. The compiler will always silently prefer the default file distributed alongside `polybench.c`.

## Steps to Reproduce
1. Create a custom PAPI event list in a separate directory (e.g., `build/papi_counters.list`).
2. Attempt to compile a PolyBench kernel with PAPI support, adding `-I build/` to the compiler flags to prioritize the custom list:
   ```bash
   gcc -O3 -I build/ -I utilities/ -I linear-algebra/kernels/2mm utilities/polybench.c linear-algebra/kernels/2mm/2mm.c -DPOLYBENCH_PAPI -lpapi -o 2mm
   ```
3. Run the compiled executable.
4. The execution will attempt to track the events defined in `utilities/papi_counters.list` (like `"L1D:REPL"`), completely ignoring the custom list in the `build/` directory. If `L1D:REPL` is not supported by the hardware, the program will fail with:
   `Error in PAPI_event_name_to_code: Event does not exist`

## Root Cause
In `utilities/polybench.c` (around line 52):
```c
#ifdef POLYBENCH_PAPI
# include <papi.h>
# define POLYBENCH_MAX_NB_PAPI_COUNTERS 96
  char* _polybench_papi_eventlist[] = {
#include "papi_counters.list"
    NULL
  };
```
The `#include "papi_counters.list"` directive forces the preprocessor to search the directory of the current file (`utilities/`) first.

## Proposed Solution
Change the `#include` directive to use angle brackets. This instructs the preprocessor to search the directories specified by the `-I` compiler flags first. 

```c
// utilities/polybench.c

  char* _polybench_papi_eventlist[] = {
#include <papi_counters.list>
    NULL
  };
```

This change allows users to override the default event list by simply providing their own via `-I` without needing to modify or delete the source files in the `utilities/` directory, while preserving the existing behavior if `-I utilities/` is provided.