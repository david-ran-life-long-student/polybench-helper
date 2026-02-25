import pandas as pd

RUNS_PER_EXPERIMENT = 10


class Mutable:
    """
    This object defines a parameter of the experiments
    for example:
    Mutable([10,100,1000,10000], name=N, is_compiler_flag=False) is a runtime arg for the problem size
    Mutable([-ftree-vectorize]) is a compile time flag
    Mutable([-O0, -O2, -O3]) also works
    value range should be a list of strings for feeding to the compiler or program
    TODO : add functionality to support env vars such as opemmp num threads
    """
    def __init__(self, value_range, name=None, is_compiler_flag=True):
        if is_compiler_flag and name is None:
            self.name = value_range[0]

        self.is_compiler_flag = is_compiler_flag
        self.value_range = value_range

    def is_bool_flag(self):
        return self.is_compiler_flag and len(self.value_range) == 1


class Study:
    """
    This object model a group of experiments
    Note that the order of the runtime arg mutables at init time are the order of the args
    // TODO : add ability to use more complex build systems than one line compiler calls
    """
    def __init__(self, build_dir, experimental_params, base_compiler_command, compiler="clang", result_parser_func=None):
        self.build_dir = build_dir
        self.compiler = compiler
        self.base_compiler_flags = base_compiler_command  # files + default flags
        self.result_parser_func = result_parser_func

        # TODO : sort the input so the order in deterministic
        self.compiler_bool_flags = [each for each in experimental_params if each.is_bool_flag()]
        self.compiler_non_bool_flags = [each for each in experimental_params if each.is_compiler_flag and not each.is_bool_flag()]
        self.runtime_mutables = [each for each in experimental_params if not each.is_compiler_flag]

        # we'll use this to name executables
        self.compiler_bool_flags_fingerprint = self.get_compiler_bool_flag_fingerprint(self.compiler_non_bool_flags)
        self.compiler_non_bool_flags_fingerprint = self.get_compiler_num_flag_fingerprint(self.compiler_non_bool_flags_fingerprint)

        self.result_df = pd.DataFrame()

    def make_empty_dataframe(self):
        """
        get an empty dataframe with the right columns based of the list of experimental param names
        :return:
        """

    def ensure_all_builds_exist(self):
        """
        make sure all permutations of the flags are built
        should check build dir for cached builds
        an executable should have the following name structure:
        <self.compiler_bool_flags_fingerprint>-<bool-flag-status-as-bits>-<non-bool-flag-fingerprint>-<value1>-<value2>...
        :return:
        """

        # make a list of all possible combinations in terms of build flag values


        # get the executable name based on the permutations

        # mutate the base command to build the executable if one doesn't exist in the build dir


    def run_experiments(self):
        """
        for each configuration (build + runtime) run RUNS_PER_EXPERIMENT times
        record all data into the pandas dataframe
        the data should be returned by the executable,
        may need some parsing, user can supply a function for doing so, we also have a default
        :return:
        """

    @staticmethod
    def default_result_parser(input_str):
        return float(input_str)

    @staticmethod
    def get_compiler_bool_flag_fingerprint(experimental_params):
        """
        return a short hash of all the names of boolean compile tine params
        :param experimental_params:
        :return:
        """

    @staticmethod
    def get_compiler_num_flag_fingerprint(experimental_params):
        """
        return a short hash of all the names of non-boolean compile tine params
        :param experimental_params:
        :return:
        """