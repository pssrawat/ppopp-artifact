%{
	#include "stdio.h"
	#include "y.tab.h"
	#include "utils.hpp"
%}
%option noyywrap

DIGIT [0-9]+
ID [a-zA-Z][a-zA-Z0-9_]*
%%

coefficient {
	return COEFFICIENT;
}

parameter {
	return PARAMETER;
}

function {
	return FUNCTION;
}

temporary {
	return TEMPORARY;
}

unroll {
	return UNROLL;
}

iterator {
	return ITERATOR;
}

reglimit {
	return REGLIMIT;
}

bool {
	yylval.ival = BOOL;
	return DATATYPE;
}

float {
    yylval.ival = FLOAT;
	return DATATYPE;
}

double {
	yylval.ival = DOUBLE;
	return DATATYPE;
}

int {
    yylval.ival = INT;
	return DATATYPE;
}

true {
	yylval.bval = true;
	return TRUE;
}

false {
	yylval.bval = false;
	return FALSE;
}

{ID} {
	yylval.str = strdup (yytext);
	return ID;
}

{DIGIT} {
	yylval.ival = atoi(yytext);
	return T_INT;
}

{DIGIT}"."{DIGIT}"f" {
    yylval.fval = atof(yytext);
    return T_FLOAT;
}

{DIGIT}"."{DIGIT}"F" {
    yylval.fval = atof(yytext);
    return T_FLOAT;
}

{DIGIT}"f" {
    yylval.fval = atof(yytext);
    return T_FLOAT;
}

{DIGIT}"F" {
    yylval.fval = atof(yytext);
    return T_FLOAT;
}

{DIGIT}"."{DIGIT} {
    yylval.dval = atof(yytext);
    return T_DOUBLE;
}

{DIGIT}"e"{DIGIT} {
	yylval.dval = atof(yytext);
	return T_DOUBLE;
}

{DIGIT}"E"{DIGIT} {
	yylval.dval = atof(yytext);
	return T_DOUBLE;
}

";" |
"+" |
"(" |
")" |
"{" |
"}" |
"|" |
"[" |
"]" |
"&" |
"%" |
"^" |
"," |
":" |
"-" |
">" |
"<" |
"=" |
"*" |
"/" {
	return yytext[0];
}

"<=" {
	return LEQ;
}

">=" {
	return GEQ;
}

"==" {
	return EQ;
}

"!=" {
	return NEQ;
}

"+=" {
	return PLUSEQ;
}

"-=" {
	return MINUSEQ;
}

"*=" {
	return MULTEQ;
}

"/=" {
	return DIVEQ;
}

"|=" {
	return OREQ;
}

"&=" {
	return ANDEQ;
}

".." {
	return DDOTS;
}

"//" {
	return COMMENT;
}

[ \t\n]+ {}

<<EOF>> {
	return 0;
}
%%
