#ifndef __CODEGEN_HPP__
#define __CODEGEN_HPP__
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <tuple>
#include <algorithm>
#include "funcdefn.hpp"

class codegen {
	private:
		start_node *start;
		std::stringstream header;
		std::stringstream gpu_code;
		std::stringstream host_code;
	public:
		codegen (start_node *);
		void print_parameters (void);
		void print_temp_decls (void);
		void print_unroll_decls (void);
		void print_var_decls (void);
		void print_array_decls (void);
		void print_func_calls (void);
		void generate_code (std::stringstream &, std::stringstream &, std::map<std::string, int> &, DATA_TYPE, int, bool, bool, bool);
};

inline codegen::codegen (start_node *node) {
	start = node;
}

#endif
