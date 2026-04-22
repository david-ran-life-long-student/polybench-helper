#include <stdio.h>

int main(int argc, char *argv[]) {
    // Read the counters from the list created by HWCounterStudy
    const char* events[] = {
#include "papi_counters.list"
        NULL
    };
    
    int num_events = 0;
    while (events[num_events] != NULL) {
        num_events++;
    }
    
    // Output fake space-separated integers for each counter.
    // The dictionary will map them based on alphabetical order 
    // as defined in HWCounterStudy counter_list.
    for(int i=0; i<num_events; i++) {
        // We'll just generate simple values like 100, 200, 300
        printf("%d ", (i+1)*100); 
    }
    printf("\n");
    return 0;
}