# Experiment Helper Framework Design

The `experiment_helper.py` framework is designed to provide a **clean, declarative interface** for running complex hardware performance studies. Instead of writing nested `for` loops to test different problem sizes, compiler flags, and parsing outputs manually, a user can declare the parameter space and the desired metrics, and the framework handles the heavy lifting: permutations, parallel compilation, caching, and execution.

Here is a breakdown of how the key classes achieve this:

## 1. The `Study` Base Class (The Orchestrator)
The `Study` class is the foundational orchestrator. It takes a list of `Mutable` parameters (which can be compiler flags, runtime arguments, or environment variables) and computes the Cartesian product of all combinations. 
* **Caching & Parallelism:** It smartly hashes the parameters to create unique executable names. It checks which binaries already exist and compiles only the missing ones in parallel using Python's `concurrent.futures`.
* **Execution:** It runs the experiments in a round-robin "sweep" to mitigate systemic bias over time, automatically checkpointing results to a CSV. 
* **Extensibility:** It exposes a `result_parser_func` hook, allowing subclasses to define exactly how the `stdout` of the executable should be translated into data columns.

## 2. `HWCounterMetric` (The Clever Interface)
This class defines a specific hardware metric (like `IPC` or `L1 Miss Rate`). The clever part is how it uses Python's `inspect` module to dynamically extract the necessary PAPI events from the function signature itself.

When a user defines a metric like this:
```python
def l1_miss_rate(PAPI_L1_DCM, PAPI_LST_INS):
    return (PAPI_L1_DCM / PAPI_LST_INS) * 100 if PAPI_LST_INS > 0 else 0

metric = HWCounterMetric("%L1m", l1_miss_rate)
```
The `HWCounterMetric` class inspects the arguments of `l1_miss_rate`. It automatically determines that this metric requires exactly `"PAPI_L1_DCM"` and `"PAPI_LST_INS"`. 

This creates a beautiful declarative interface: the user just writes the math they want, using the PAPI counter names as variables. They don't have to maintain a separate list of string names or map array indices to variables.

## 3. `HWCounterStudy` (The PAPI Injector)
`HWCounterStudy` subclasses `Study` to specifically handle Polybench programs instrumented with PAPI hardware counters.

* **Automated Setup:** It takes a list of `HWCounterMetric` objects. It iterates through all of them, aggregates the unique set of required PAPI counters (extracted via the `inspect` trick above), and dynamically generates a `papi_counters.list` file in the build directory. 
* **Compiler Injection:** It intercepts the base compiler command and injects `-I {build_dir}` and `-DPOLYBENCH_PAPI` so that the C compiler picks up the generated list of counters.
* **Custom Parsing:** It overrides the `result_parser_func`. When the C program prints a space-separated list of integers, the parser maps those integers back to the sorted list of PAPI counter names. It then unpacks those mapped values as `**kwargs` directly into the compute functions defined in the `HWCounterMetric` objects, generating the final calculated ratios for the CSV.

**The Result:** The user gets to write highly readable Python defining *what* they want to test and compute, while the framework handles *how* to orchestrate the compiler, the PAPI runtime linkage, and the data aggregation.
