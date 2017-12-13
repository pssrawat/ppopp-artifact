%{
	#include <cstdio>
	#include <string>
	#include <vector>
	#include "grammar.hpp"

	extern FILE *yyin;
	int yylex (void);
	int yyparse (void);
	void yyerror(const char *str) {
		fprintf(stderr,"error: %s\n",str);
	}
	void grammar::set_input (FILE *in) {
	    yyin = in;
	}
	void grammar::parse () {
	    do {
	        yyparse ();
	    } while (!feof (yyin));
	}
%}

%union{
	int ival;
	double dval;
	float fval;
	bool bval;
	char *str;
	class start_node *startnode;
	class funcdefn *func_defn;
	class stmtlist *stmt_list;
	class expr_node *exprnode;
	class shiftvec_node *shiftvecnode;
	class string_list *stringlist;
	class range_list *rangelist;
	class array_decl *arraydecl;
}

%token <str> ID
%token <ival> DATATYPE
%token <ival> TRUE
%token <ival> FALSE
%token <ival> T_INT
%token <fval> T_FLOAT
%token <dval> T_DOUBLE
%token PARAMETER FUNCTION TEMPORARY COEFFICIENT ITERATOR UNROLL REGLIMIT LEQ GEQ EQ NEQ PLUSEQ MINUSEQ MULTEQ DIVEQ ANDEQ OREQ DDOTS COMMENT

/* Associativity */
%left '|'
%left '&'
%left EQ NEQ
%left '<' LEQ '>' GEQ
%left '+' '-'
%left '*' '/' '%'
%left UMINUS UPLUS
%left '^'

%type <startnode> program
%type <func_defn> funcfn
%type <stmt_list> pointstmts 
%type <exprnode> stmt
%type <shiftvecnode> offsetvar
%type <shiftvecnode> offsetlist
%type <exprnode> arrayaccess 
%%

start : {grammar::start = new start_node ();} program {}
	;

program : funcfn {} 
	;

funcfn : pointstmts {funcdefn *node = new funcdefn ($1);
					grammar::start->push_func_defn (node);}
	;

/* A list of statements describing accesses at generic point, of the form LHS = RHS */
pointstmts : pointstmts stmt '=' stmt ';' {stmtnode *node = new stmtnode (ST_EQ, $2, $4);
					$1->push_stmt (node);
					$$ = $1;}
	| pointstmts stmt PLUSEQ stmt ';' {stmtnode *node = new stmtnode (ST_PLUSEQ, $2, $4);
					$1->push_stmt (node);
					$$ = $1;}
	| pointstmts stmt MINUSEQ stmt ';' {stmtnode *node = new stmtnode (ST_MINUSEQ, $2, $4);
					$1->push_stmt (node);
					$$ = $1;}
	| pointstmts stmt MULTEQ stmt ';' {stmtnode *node = new stmtnode (ST_MULTEQ, $2, $4);
					$1->push_stmt (node);
					$$ = $1;}
	| pointstmts stmt DIVEQ stmt ';' {stmtnode *node = new stmtnode (ST_DIVEQ, $2, $4);
					$1->push_stmt (node);
					$$ = $1;}
	| pointstmts stmt ANDEQ stmt ';' {stmtnode *node = new stmtnode (ST_ANDEQ, $2, $4);
					$1->push_stmt (node);
					$$ = $1;}
	| pointstmts stmt OREQ stmt ';' {stmtnode *node = new stmtnode (ST_OREQ, $2, $4);
					$1->push_stmt (node);
					$$ = $1;}
	/* Add support for comments */
	| pointstmts COMMENT stmt '=' stmt ';' {$$ = $1;}
	| pointstmts COMMENT stmt PLUSEQ stmt ';' {$$ = $1;}
	| pointstmts COMMENT stmt MINUSEQ stmt ';' {$$ = $1;}
	| pointstmts COMMENT stmt MULTEQ stmt ';' {$$ = $1;}
	| pointstmts COMMENT stmt DIVEQ stmt ';' {$$ = $1;}
	| pointstmts COMMENT stmt ANDEQ stmt ';' {$$ = $1;}
	| pointstmts COMMENT stmt OREQ stmt ';' {$$ = $1;}

	| {stmtlist *node = new stmtlist (); 
	$$ = node;}
	;

/* A single statement of the form (B[k+2,j+1,i+3] + 2.0f*C[k+1,j+3,i+2])*/
stmt : ID {$$ = new id_node ($1);}
	| offsetvar {$$ = $1;}
	| TRUE {$$ = new datatype_node<bool> ($1, BOOL);}
	| FALSE {$$ = new datatype_node<bool> ($1, BOOL);}
	| T_INT {$$ = new datatype_node<int> ($1, INT);}
	| T_DOUBLE {$$ = new datatype_node<double> ($1, DOUBLE);}
	| T_FLOAT {$$ = new datatype_node<float> ($1, FLOAT);}
	| stmt '|' stmt {$$ = new binary_node (T_OR, $1, $3);} 
	| stmt '&' stmt {$$ = new binary_node (T_AND, $1, $3);} 
	| stmt EQ stmt {$$ = new binary_node (T_EQ, $1, $3);}
	| stmt NEQ stmt {$$ = new binary_node (T_NEQ, $1, $3);}
	| stmt '<' stmt {$$ = new binary_node (T_LT, $1, $3);}
	| stmt LEQ stmt {$$ = new binary_node (T_LEQ, $1, $3);}
	| stmt '>' stmt {$$ = new binary_node (T_GT, $1, $3);}
	| stmt GEQ stmt {$$ = new binary_node (T_GEQ, $1, $3);}
	| stmt '+' stmt {$$ = new binary_node (T_PLUS, $1, $3);}  
	| stmt '-' stmt {$$ = new binary_node (T_MINUS, $1, $3);}
	| stmt '*' stmt {$$ = new binary_node (T_MULT, $1, $3);}
	| stmt '/' stmt {$$ = new binary_node (T_DIV, $1, $3);}
	| stmt '%' stmt {$$ = new binary_node (T_MOD, $1, $3);}
	| '-' stmt %prec UMINUS {$$ = new uminus_node ($2);} 
	| stmt '^' stmt {$$ = new binary_node (T_EXP, $1, $3);}
	| '(' stmt ')' {$2->set_nested (); 
			$$ = $2;}
	| ID '(' stmt ')' {$$ = new function_node ($1, $3);} 
	;

/* Array access, of the form A[k+1][j+2][i+1] or A[k+1,j+2,i+1] */
offsetvar : ID '[' offsetlist ']' {$3->set_name ($1); 
				$$ = $3;}
	;

arrayaccess : ID {$$ = new id_node ($1);}
	| T_INT {$$ = new datatype_node<int> ($1, INT);}
	| arrayaccess '+' arrayaccess {$$ = new binary_node (T_PLUS, $1, $3);}  
	| arrayaccess '-' arrayaccess {$$ = new binary_node (T_MINUS, $1, $3);}
	| arrayaccess '*' arrayaccess {$$ = new binary_node (T_MULT, $1, $3);}
	| arrayaccess '/' arrayaccess {$$ = new binary_node (T_DIV, $1, $3);}
	| '-' arrayaccess %prec UMINUS {$$ = new uminus_node ($2);}
	;

offsetlist : offsetlist ',' arrayaccess {$1->push_index ($3); 
					$$ = $1;} 
	| offsetlist ']''[' arrayaccess {$1->push_index ($4); 
					$$ = $1;} 
	| arrayaccess {shiftvec_node *node = new shiftvec_node ();
		node->push_index ($1); 
		$$ = node;}
	;

%%
