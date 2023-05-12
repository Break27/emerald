use std::ops::{Deref, Not};
use std::str::FromStr;
use regex_macro::regex;
use crate::parser::error::LexError;
use crate::parser::error::LexError::*;
use crate::parser::lexer::Token::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Identifier(String),
    Operator(String),
    Atom(String),
    FmtString(Vec<String>, Vec<Vec<Token>>),
    StringLiteral(String),
    Integer(i32),
    Float(f64),

    By,
    Fn,
    Do,
    Return,
    Begin,
    Final,
    End,
    Module,
    Import,
    Trait,

    If,
    Elif,
    Else,
    For,
    While,

    Is,
    In,
    And,
    Or,
    Not,

    OpeningParenthesis,
    ClosingParenthesis,
    OpeningCurlyBrace,
    ClosingCurlyBrace,
    OpeningBracket,
    ClosingBracket,
    RangeInclusive,
    RangeExclusive,

    Dot,
    Comma,
    Chain,
    Delimiter,
    Apostrophe,
    Backquote,

    Annotator,
    Decorator,
    Assign,
    Modify,
    Arrow,
}

macro_rules! match_capture {
    ($caps:ident, $pat:pat, { $name:expr => $mat:expr $(,$o_name:expr => $o_mat:expr)* }) => {
        if let Some(m) = $caps.name($name) {
            match m.as_str() {
                $pat => $mat
            }
        }
        $(
            else if let Some(m) = $caps.name($o_name) {
                match m.as_str() {
                    $pat => $o_mat
                }
            }
        )*
        else {
            return Err(UnrecognizedTokenError {
                token: $caps[0].to_string()
            })
        }
    };
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, LexError> {
    let comments = regex!(r"(?m)#.*\n");
    let code = comments.replace_all(input, "\n");

    let mut result = Vec::new();

    let token_regex = regex!(concat!(
        r"(?P<identifier>([\$~%_]|\p{L})+\d*)|",
        r"(?P<atom>:[\p{Alphabetic}_]\w*)|",
        r"(?P<float>(\d*\.\d+)+)|",
        r"(?P<integer>\d+)|",
        r"(?P<operator>[\+\-\*\^\|\\@&=/<>?!:]+)|",
        r"(?P<enclosure>[\(\)\[\]\{\}])|",
        r"(?P<delimiter>[\n,;])|",
        r"(?P<dot>\.+)|",
         "(?P<string>('[^']*')|(\"[^\"]*\"))|",
        r"(\S+)"
    ));

    for caps in token_regex.captures_iter(code.deref()) {
        let token = match_capture!(caps, it, {
            "identifier" => match it {
                "fn"      => Fn,
                "by"      => By,
                "do"      => Do,
                "return"  => Return,
                "begin"   => Begin,
                "final"   => Final,
                "end"     => End,
                "module"  => Module,
                "import"  => Import,
                "trait"   => Trait,

                "if"    => If,
                "elif"  => Elif,
                "else"  => Else,
                "for"   => For,
                "while" => While,

                "is"  => Is,
                "in"  => In,
                "or"  => Or,
                "and" => And,
                "not" => Not,
                
                "nil" | "true" | "false" => Atom(it.to_string()),
                ident => Identifier(ident.to_string()),
            },
            "atom" => {
                Atom(it[1..].to_string())
            },
            "integer" => {
                Integer(parse_number(it)?)
            },
            "float" => {
                Float(parse_number(it)?)
            },
            "string" => {
                parse_string(it)?
            },
            "operator" => match it {
                "@" => Decorator,
                ":" => Annotator,
                "=" => Assign,
                ":=" => Modify,
                "->" => Arrow,
                "::" => Chain,
                 __  => Operator(it.to_string())
            },
            "enclosure" => match it {
                "(" => OpeningParenthesis,
                ")" => ClosingParenthesis,
                "{" => OpeningCurlyBrace,
                "}" => ClosingCurlyBrace,
                "[" => OpeningBracket,
                 _  => ClosingBracket
            },
            "dot" => match it.len() {
                1 => Dot,
                2 => RangeExclusive,
                3 => RangeInclusive,

                _ => return Err(UnrecognizedTokenError {
                    token: it.to_string()
                })
            },
            "delimiter" => match it {
                "," => Comma,
                 _  => Delimiter,
            }
        });

        result.push(token);
    }

    Ok(result)
}

fn parse_string(it: &str) -> Result<Token, LexError> {
    let mut text = it[1..].to_string();

    // omit character ' or " in the last
    // also if matched, try parsing string formatters
    if let '"' = text.pop().unwrap() {
        let mut mats = Vec::new();
        let mut strs = Vec::new();

        loop {
            if strs.is_empty().not() {
                text = strs.pop().unwrap();
            }

            let start = match text.find('{') {
                Some(i) => i,
                None => break
            };

            let end = match text.find('}') {
                Some(i) => i,
                None => break
            };

            let head = text.get(..start).unwrap();
            let tail = text.get((end + 1)..).unwrap();

            // range shrunken to omit '{' and '}'
            let range = (start + 1)..end;
            let slice = text.get(range).unwrap();

            let parsed = tokenize(slice)?;
            mats.push(parsed);

            strs.push(head.to_string());
            strs.push(tail.to_string());
        }

        if mats.is_empty().not() {
            // the last bit of text
            strs.push(text);

            return Ok(FmtString(strs, mats))
        }
    }

    Ok(StringLiteral(text))
}

fn parse_number<T: FromStr>(it: &str) -> Result<T, LexError> {
    it.parse().or(Err(InvalidNumberError {
        token: it.to_string()
    }))
}
