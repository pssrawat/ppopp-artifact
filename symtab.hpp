#ifndef __SYMTAB_HPP__
#define __SYMTAB_HPP__
#include <cstdio>
#include <cstring>
#include <map>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <vector>

/* A class that represents vector. Needed for parsing */
template <typename T>
class vectab {
	protected:
		std::vector<T> vec_list;
	public:
		virtual void push_back (T);
		virtual std::vector<T> get_list (void);
};

template <typename T>                                                                            
inline void vectab<T>::push_back (T value) {                                                            
	vec_list.push_back (value);                                                                  
}                                                                                                

template <typename T>                                                                            
inline std::vector<T> vectab<T>::get_list (void) {                                                      
	return vec_list;                                                                             
} 

/* A class that represents a vector of string */
class string_list : public vectab<std::string> {
	public:
		void push_back (std::string);
		std::vector<std::string> get_list (void);
};

inline void string_list::push_back (std::string value) {
	vec_list.push_back (value);
}

inline std::vector<std::string> string_list::get_list (void) {
	return vec_list;
}

/* A class that represents a vector of expressions */
class expr_list : public vectab<expr_node *> {
	public:
		void push_back (expr_node *);
		std::vector<expr_node *> get_list (void);
};

inline void expr_list::push_back (expr_node *value) {
	vec_list.push_back (value);
}

inline std::vector<expr_node*> expr_list::get_list (void) {
	return vec_list;
}

/* A class that represents symbol table. Basically a map from string to 
   any data structure. */
template <typename T>
class symtab {
	protected:
		std::map<std::string, T> symbol_list;
	public:
		void push_symbol (char *str, T);
		void delete_symbol (char *str);
		T find_symbol (char *str);	
		std::map<std::string, T> get_symbol_list (void); 
};

template <typename T>
inline std::map<std::string, T> symtab<T>::get_symbol_list (void) {
	return symbol_list;
}

template <typename T>
inline void symtab<T>::push_symbol (char *s, T value) {
	std::string key = std::string (s);
	assert (symbol_list.find (key) == symbol_list.end () && "Assigned name already exists"); 
	symbol_list.insert (make_pair (key, value));
}

template <typename T>
inline void symtab<T>::delete_symbol (char *s) {
	std::string key = std::string (s);
	symbol_list.erase (key);
}

template <typename T>
inline T symtab<T>::find_symbol (char *s) {
	std::string key = std::string (s);
	typename std::map <std::string, T>::iterator it = symbol_list.find (key);
	if (it != symbol_list.end ())
		return it->second;
	return NULL; 
}

#endif
