#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <map>
#include <cassert>
#include <boost/dynamic_bitset.hpp>
#include "datatypes.hpp"

inline DATA_TYPE get_data_type (int a) {
	DATA_TYPE ret = INT;
	switch (a) {
		case 0:
			ret = INT;
			break;
		case 1:
			ret = FLOAT;
			break;
		case 2:
			ret = DOUBLE;
			break;
		case 3:
			ret = BOOL;
			break;
		default:
			fprintf (stderr, "Data type not supported\n");
			exit (1);
	}
	return ret;
}

inline DATA_TYPE infer_data_type (DATA_TYPE a, DATA_TYPE b) {
	if (a == DOUBLE || b == DOUBLE) 
		return DOUBLE;
	if (a == FLOAT || b == FLOAT) 
		return FLOAT; 
	if (a == INT || b == INT)
		return INT;
	return BOOL;
}

inline std::string print_stmt_op (STMT_OP op_type) {
	std::string str;
	switch (op_type) {
		case ST_PLUSEQ:
			str = std::string (" += ");
			break;
		case ST_MINUSEQ:
			str = std::string (" -= ");
			break;
		case ST_MULTEQ:
			str = std::string (" *= ");
			break;
		case ST_DIVEQ:
			str = std::string (" /= ");
			break;
		case ST_EQ:
			str = std::string (" = ");
			break;
		default:
			fprintf (stderr, "Statement op type not supported\n");
			exit (1);
	}
	return str; 
}

inline std::string print_operator (OP_TYPE op, DATA_TYPE type) {
	std::string str;
	std::string tail = (type == DOUBLE) ? "pd" : "ps";
	switch (op) {
		case T_PLUS:
			str = std::string ("_mm256_add_") + tail;
			break;
		case T_MINUS:
			str = std::string ("_mm256_sub_") + tail;
			break;
		case T_MULT:
			str = std::string ("_mm256_mul_") + tail;
			break;
		case T_DIV:
			str = std::string ("_mm256_div_") + tail;
			break;
		case T_OR:
			str = std::string ("_mm256_or_") + tail;
			break;
		case T_AND:
			str = std::string ("_mm256_and_") + tail;
			break;
		default:
			fprintf (stderr, "Operator not supported\n");
			exit (1);
	}
	return str;
}

inline std::string print_operator (OP_TYPE op) {
	std::string str;
	switch (op) {
		case T_PLUS:
			str = std::string (" + ");
			break;
		case T_MINUS:
			str = std::string (" - ");
			break;
		case T_MULT:
			str = std::string (" * ");
			break;
		case T_DIV:
			str = std::string (" / ");
			break;
		case T_MOD:
			str = std::string (" % ");
			break;
		case T_EXP:
			str = std::string (" ^ ");
			break;
		case T_LEQ:
			str = std::string (" <= ");
			break;
		case T_GEQ:
			str = std::string (" >= ");
			break;
		case T_NEQ:
			str = std::string (" != ");
			break;
		case T_EQ:
			str = std::string (" == ");
			break;
		case T_LT:
			str = std::string (" < ");
			break;
		case T_GT:
			str = std::string (" > ");
			break;
		case T_OR:
			str = std::string (" | ");
			break;
		case T_AND:
			str = std::string (" & ");
			break;
		default:
			fprintf (stderr, "Operator not supported\n");
			exit (1);
	}
	return str;
}

inline std::string print_data_type (DATA_TYPE a) {
	std::string ret;
	switch (a) {
		case INT:
			ret = std::string ("int ");
			break;
		case FLOAT:
			ret = std::string ("float ");
			break;
		case DOUBLE:
			ret = std::string ("double ");
			break;
		case BOOL: 
			ret = std::string ("bool ");
			break;
		default:
			fprintf (stderr, "Data type not supported\n");
			exit (1);
	}
	return ret;
}

inline STMT_OP convert_op_to_stmt_op (OP_TYPE a) {
	STMT_OP ret = ST_EQ;
	switch (a) {
		case T_PLUS:
			ret = ST_PLUSEQ;
			break;
		case T_MINUS:
			ret = ST_MINUSEQ;
			break;
		case T_MULT:
			ret = ST_MULTEQ;
			break;
		case T_AND:
			ret = ST_ANDEQ;
			break;
		case T_OR:
			ret = ST_OREQ;
			break;
		case T_EQ:
			ret = ST_EQ;
			break;
		default:
			fprintf (stderr, "OP_TYPE not supported\n");
			exit (1);
	}
	return ret;
}

inline STMT_OP distributive_stmt_op (STMT_OP a, STMT_OP b) {
	// If a is assignment op, just follow b
	if (a == ST_EQ) 
		return b;
	STMT_OP ret;
	if (a == ST_PLUSEQ) {
		switch (b) {
			case ST_PLUSEQ:
				ret = ST_PLUSEQ;
				break;
			case ST_EQ:
				ret = ST_PLUSEQ;
				break;
			case ST_MINUSEQ:
				ret = ST_MINUSEQ;
				break;
			default:
				fprintf (stderr, "distributivity for STMT_OP b not supported\n");
				exit (1);
		}
	}
	else if (a == ST_MINUSEQ) {
		switch (b) {
			case ST_PLUSEQ:
				ret = ST_MINUSEQ;
				break;
			case ST_EQ:
				ret = ST_MINUSEQ;
				break;
			case ST_MINUSEQ:
				ret = ST_PLUSEQ;
				break;
			default:
				fprintf (stderr, "distributivity for STMT_OP b not supported\n");
				exit (1);
		}
	}
	else {
		fprintf (stderr, "distributivity for STMT_OP not supported\n");
		exit (1);
	}
	return ret;
}


inline OP_TYPE convert_stmt_op_to_op (STMT_OP a) {
	OP_TYPE ret;
	switch (a) {
		case ST_PLUSEQ:
			ret = T_PLUS;
			break;
		case ST_MINUSEQ:
			ret = T_MINUS;
			break;
		case ST_MULTEQ:
			ret = T_MULT;
			break;
		case ST_ANDEQ:
			ret = T_AND;
			break;
		case ST_OREQ:
			ret = T_OR;
			break;
		case ST_EQ:
			ret = T_EQ;
			break;
		default:
			fprintf (stderr, "STMT_OP not supported\n");
			exit (1);
	}
	return ret;
}

inline int get_init_val (STMT_OP a) {
	int ret = 0;
	switch (a) {
		case ST_PLUSEQ:
			ret = 0;
			break;
		case ST_MINUSEQ:
			ret = 0;
			break;
		case ST_MULTEQ:
			ret = 1;
			break;
		case ST_DIVEQ:
			ret = 1;
			break;
		case ST_ANDEQ:
			ret = 1;
			break;
		case ST_OREQ:
			ret = 0;
			break;
		case ST_EQ:
			ret = 0;
			break;
		default:
			fprintf (stderr, "Operation not associative\n");
			exit (1);
	}
	return ret;
}

inline STMT_OP get_cur_op (STMT_OP a, bool flip) {
	STMT_OP ret = a;
	switch (a) {
		case ST_PLUSEQ:
			ret = flip ? ST_MINUSEQ : ST_PLUSEQ;
			break;
		case ST_MINUSEQ:
			ret = flip ? ST_PLUSEQ : ST_MINUSEQ;
			break;
		default:
			break;
	}
	return ret;
}

inline STMT_OP acc_start_op (STMT_OP a) {
	STMT_OP ret = a;
	switch (a) {
		case ST_MINUSEQ:
			ret = ST_PLUSEQ;
			break;
		case ST_DIVEQ:
			ret = ST_MULTEQ;
			break;
		default:
			break;
	}
	return ret;
}

inline void addArrays (unsigned int *arr1, unsigned int *arr2, int size) {
	for (int i=0; i<size; i++) arr1[i] += arr2[i];
}

inline void subtractArrays (unsigned int *arr1, unsigned int *arr2, int size) {
	for (int i=0; i<size; i++) arr1[i] -= arr2[i];
}

inline void copyArray (unsigned int *arr1, unsigned int *arr2, int size) {
	std::copy (arr2, arr2+size, arr1);
}

inline void resetArray (unsigned int *arr1, int size) {
	for (int i=0; i<size; i++) arr1[i] = 0;
}

inline void print_bitset (std::map<std::string, int> string_to_index, boost::dynamic_bitset<> used_labels, unsigned int *use_frequency, int label_count) {
	// Reverse the map
	std::map<int, std::string> index_to_string;
	for (std::map<std::string,int>::iterator it=string_to_index.begin(); it!=string_to_index.end(); it++) 
		index_to_string[it->second] = it->first;
	for (int i=0; i<label_count; i++) {
		if (used_labels[i] == true)  
			printf ("(%s, %u)  ", (index_to_string[i]).c_str(), use_frequency[i]); 
		else 
			assert (use_frequency[i] == 0 && "Use frequency non-zero for an unused label (print_bitset)"); 
	}
	std::cout << std::endl;
}

inline void print_bitset (std::map<std::string, int> string_to_index, boost::dynamic_bitset<> used_labels, int label_count) {
	// Reverse the map
	std::map<int, std::string> index_to_string;
	for (std::map<std::string,int>::iterator it=string_to_index.begin(); it!=string_to_index.end(); it++) 
		index_to_string[it->second] = it->first;
	for (int i=0; i<label_count; i++) 
		if (used_labels[i] == true) printf ("%s  ", (index_to_string[i]).c_str());
	std::cout << std::endl;
}

inline void print_frequency (std::map<std::string, int> string_to_index, unsigned int *frequency, int label_count) {
	// Reverse the map
	std::map<int, std::string> index_to_string;
	for (std::map<std::string,int>::iterator it=string_to_index.begin(); it!=string_to_index.end(); it++) 
		index_to_string[it->second] = it->first;
	for (int i=0; i<label_count; i++) 
		if (frequency[i] > 0) printf ("(%s, %u)  ", (index_to_string[i]).c_str(), frequency[i]);
	std::cout << std::endl;
}

#endif
