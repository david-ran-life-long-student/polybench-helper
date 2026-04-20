#include <stdio.h>

int main(int argc, char *argv[]) {
    fprintf(stderr, "=== Program Execution Start ===\n");
    fprintf(stderr, "Compile-time macros:\n");
#ifdef TEST_MACRO
    fprintf(stderr, "  TEST_MACRO is DEFINED\n");
#else
    fprintf(stderr, "  TEST_MACRO is NOT DEFINED\n");
#endif

    fprintf(stderr, "Optimization Level (from __OPTIMIZE__ macro):\n");
#ifdef __OPTIMIZE__
    fprintf(stderr, "  Optimizations ENABLED\n");
#else
    fprintf(stderr, "  Optimizations DISABLED\n");
#endif

    fprintf(stderr, "Runtime arguments (argc=%d):\n", argc);
    for (int i = 0; i < argc; i++) {
        fprintf(stderr, "  argv[%d] = %s\n", i, argv[i]);
    }
    fprintf(stderr, "=== Program Execution End ===\n\n");
    
    // Output a dummy performance result for the parser
    printf("42.0\n");
    return 0;
}