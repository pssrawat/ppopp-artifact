#ifndef __VARDECL_HPP__
#define __VARDECL_HPP__
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include "utils.hpp"
#include "exprnode.hpp"
#include "symtab.hpp"

class array_range {
	private:
		expr_node *lo;
		expr_node *hi;
	public:
		array_range (expr_node *, expr_node *);
		expr_node *get_lo_range ();
		expr_node *get_hi_range ();
};

inline array_range::array_range (expr_node *lo_id, expr_node *hi_id) {
	lo = lo_id;
	hi = hi_id;
}

inline expr_node * array_range::get_lo_range (void) {
	return lo;
}

inline expr_node * array_range::get_hi_range (void) {
	return hi;
}

class range_list : public vectab<array_range *> {
	public:
		void push_back (array_range *);
		std::vector<array_range *> get_list (void);
};

inline void range_list::push_back (array_range *value) {
	vec_list.push_back (value);
}

inline std::vector<array_range *> range_list::get_list (void) {
	return vec_list;
}

class array_decl {
	private:
		range_list *range;
		std::string name;
		DATA_TYPE data_type;
	public:
		array_decl (DATA_TYPE, char *, range_list *);
		void push_range (array_range *);
		std::vector<array_range *> get_array_range (void);
		range_list *get_range_list (void);
		std::string get_array_name (void);
		DATA_TYPE get_array_type (void);
};

inline array_decl::array_decl (DATA_TYPE t, char *str, range_list *r) {
	data_type = t;
	range= r;
	name = std::string (str);
}

inline void array_decl::push_range (array_range *r) {
	range->push_back (r);
}

inline DATA_TYPE array_decl::get_array_type (void) {
	return data_type;
}

inline std::vector<array_range *> array_decl::get_array_range (void) {
	return range->get_list ();
}

inline range_list *array_decl::get_range_list (void) {
	return range;
}

inline std::string array_decl::get_array_name (void) {
	return name;
}

class func_call {
	private:
		std::string func_name;
		string_list *out_list, *args;
	public:
		func_call (char *, string_list *, string_list *);
		func_call (char *, string_list *);
		void set_name (char *);
		std::string get_name (void);
		void set_out_list (string_list *);
		void push_arg (char *);
		void push_out_var (char *);
		std::vector<std::string> get_out_list (void);
		std::vector<std::string> get_arg_list (void);
};

inline func_call::func_call (char *s, string_list *arg) {
	func_name = std::string (s);
	args = arg;
	out_list = new string_list ();
}

inline func_call::func_call (char *s, string_list *arg, string_list *out) {
	func_name = std::string (s);
	out_list = out;
	args = arg;
}

inline void func_call::set_name (char *s) {
	func_name = std::string (s);
}

inline std::string func_call::get_name (void) {
	return func_name;
}

inline void func_call::set_out_list (string_list *s) {
	out_list = s;
}

inline void func_call::push_arg (char *s) {
	args->push_back (std::string (s));
}

inline void func_call::push_out_var (char *s) {
	out_list->push_back (std::string (s));
}

inline std::vector<std::string> func_call::get_out_list (void) {
	return out_list->get_list ();
}

inline std::vector<std::string> func_call::get_arg_list (void) {
	return args->get_list ();
}

#endif
