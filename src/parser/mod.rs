use SnippetParsingResult::*;
use crate::parser::error::ParseError;
use crate::parser::ast::{ASTNode, Expression, Expression::*, Prototype, Type};
use crate::parser::error::ParseError::*;
use crate::parser::lexer::Token;
use crate::parser::lexer::Token::*;
use crate::parser::config::ParserConfiguration as Config;

pub mod ast;
pub mod lexer;
pub mod error;
pub mod config;

pub type ParsingResult = Result<(Vec<ASTNode>, Vec<Token>), (ParseError, Token)>;
type ExpressionResult = SnippetParsingResult<Expression>;

pub enum SnippetParsingResult<T> {
    Success(T, Vec<Token>),
    Incomplete,
    Failure(ParseError)
}

/// try parsing tokens
macro_rules! parse {
    ($parsed_tokens:expr; $tokens:expr, $config:ident;
        $primary:ident(); $( $function:ident($($arg:expr),*) );+
    ) => {{
        let mut expr = parse! {
            $parsed_tokens;
            $primary($tokens, $config)
        };
        $(
            expr = parse! {
                $parsed_tokens;
                $function($tokens, $config $(,$arg)*, expr)
            };
        )+
        expr
    }};

    ($parsed_tokens:expr; $function:ident($tokens:expr, $config:ident $(,$arg:expr)*)) => (
        match $function($tokens, $config, $($arg),*) {
            Success(ast, _tokens) => {
                $parsed_tokens.extend(_tokens.into_iter());
                ast
            },
            Incomplete => {
                $parsed_tokens.reverse();
                $tokens.extend($parsed_tokens.into_iter());
                return Incomplete
            },
            Failure(e) => return Failure(e)
        }
    );
}

/// Expect for a token. If match, eat the token and push it to the parsed tokens.
/// Otherwise, execute $not_matched statement / block, or return parsing failure
/// by default.
macro_rules! expect {
    ($tokens:ident, $parsed_tokens:ident, {
        $($pattern:pat $(if $cond:expr)? => $result:stmt,)+ $(? => $not_matched:stmt)?
    }) => {
        match $tokens.last().map(|i| { i.clone() }) {
            $(
                #[allow(unused_parens)]
                Some($pattern) $(if $cond)? => {
                    let token = $tokens.pop();
                    $parsed_tokens.push(token.unwrap());
                    $result
                },
            )+
            $(
                _ => { $not_matched }
            )?
            #[allow(unreachable_patterns)]
            _ => return Failure(MissingExpectedTokenError {
                tokens: vec![$( stringify!($pattern).to_string() ),+]
            })
        }
    }
}

/// expect and parse an enclosure.
macro_rules! enclosure {
    ($list:ident, $tokens:ident, $parsed_tokens:ident, {
        ($($closing:tt)+) $(|$any:pat)? => $parse:expr,
        $($delimiter:tt)|+ => continue
    }) => {{
        let mut index = $list.len();
        loop {
            index += 1;

            expect!($tokens, $parsed_tokens, {
                $($closing)+ => break,
                ? => ()
            });

            if index % 2 == 0 {
                expect!($tokens, $parsed_tokens, {
                    $( $delimiter => continue, )+
                    $( ? => { stringify!($any); break } )?
                });
            }

            $list.push($parse);
        }
    }}
}

pub fn parse(tokens: &[Token], parsed_tree: &[ASTNode], config: &Config) -> ParsingResult {
    // we read tokens from the end of the vector
    // using it as a stack
    let mut stack = tokens.to_vec();
    stack.reverse();

    let mut ast = parsed_tree.to_vec();

    loop {
        // look at the current token and determine what to parse
        // based on its value
        let token = match stack.last() {
            Some(Delimiter) => {
                stack.pop();
                continue
            },
            Some(token) => token.clone(),
            _ => break
        };
        
        match parse_node(&mut stack, config) {
            Success(node, _) => ast.push(node),
            Incomplete => break,
            Failure(e) => return Err((e, token))
        }
    }

    stack.reverse();
    Ok((ast, stack))
}

fn parse_node(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    match tokens.last() {
        Some(Begin) => parse_begin_block(tokens, config),
        Some(Module) => parse_module(tokens, config),
        Some(Return) => parse_return(tokens, config),
        Some(Import) => parse_import(tokens, config),
        Some(Decorator) => parse_decoration(tokens, config),
        Some(Fn) => parse_prototype(tokens, config),
        Some(_) => parse_expression(tokens, config),
        None => Incomplete
    }
}

fn parse_primary_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    match tokens.last() {
        Some(By) => parse_scope_expr(tokens, config),
        Some(Atom(_)) => parse_atom_expr(tokens, config),
        Some(Identifier(_)) => parse_basic_expr(tokens, config),
        Some(Integer(_) | Float(_)) => parse_number_literal_expr(tokens, config),
        Some(StringLiteral(_)) => parse_string_literal_expr(tokens, config),
        Some(FmtString(_, _)) => parse_formatted_string_expr(tokens, config),
        Some(OpeningCurlyBrace) => parse_block_expr(tokens, config),
        Some(OpeningParenthesis | OpeningBracket) => parse_enclosure_expr(tokens, config),
        Some(Delimiter) => { tokens.pop(); Incomplete },

        Some(e) => Failure(UnexpectedTokenError { token: format!("{:?}", e) }),
        None => Incomplete,
    }
}

fn parse_enclosure_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();

    let mut list = Vec::new();
    let result;
    expect!(tokens, parsed_tokens, {
        OpeningParenthesis => {
            enclosure!(list, tokens, parsed_tokens, {
                (ClosingParenthesis) => parse! {
                    parsed_tokens; tokens, config;
                    parse_primary_expr();
                    parse_binary_expr(0);
                    parse_tail_expr()
                },
                Comma | Delimiter => continue
            });

            result = GroupExpr(list);
        },
        OpeningBracket => {
            enclosure!(list, tokens, parsed_tokens, {
                (ClosingBracket) => parse! {
                    parsed_tokens; tokens, config;
                    parse_primary_expr();
                    parse_binary_expr(0);
                    parse_tail_expr()
                },
                Comma | Delimiter => continue
            });

            result = ListExpr(list);
        },
    });

    Success(result, parsed_tokens)
}

fn parse_block_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();
    parsed_tokens.push(tokens.pop().unwrap());

    let mut args = Vec::new();
    expect!(tokens, parsed_tokens, {
        Operator(x) if x.eq("|") =>
            enclosure!(args, tokens, parsed_tokens, {
                (Operator(x) if x.eq("|")) => parse! {
                    parsed_tokens;
                    parse_annotated_value(tokens, config, String::new())
                },
                Comma | Delimiter => continue
            }),
        Operator(x) if x.eq("||") => {/* empty parameters */},
        ? => ()
    });

    // return type
    expect!(tokens, parsed_tokens, {
        Arrow => {
            let ty = parse! {
                parsed_tokens;
                parse_type(tokens, config)
            };

            // todo - type system
        },
        ? => ()
    });

    // body - only expressions are allowed
    let mut body = Vec::new();
    enclosure!(body, tokens, parsed_tokens, {
        (ClosingCurlyBrace) => parse! {
            parsed_tokens;
            parse_node(tokens, config)
        },
        Delimiter => continue
    });

    let result = BlockExpr {
        args,
        body,
    };

    Success(result, parsed_tokens)
}

fn parse_binary_expr(tokens: &mut Vec<Token>, config: &Config, precedence: i32, lhs: Expression) -> ExpressionResult {
    // start with LHS value
    let mut result = lhs.clone();
    let mut parsed_tokens = Vec::new();

    macro_rules! match_operator {
        ($pattern:pat if $cond:expr => $matched:expr) => (
            match tokens.last().map(|i| i.clone()) {

                Some(Operator(ref name)) => {
                    match config.operators.get(name) {
                        $pattern if $cond => {
                            (name.to_string(), $matched)
                        },
                        None => return Failure(UndefinedOperatorError {
                            name: name.to_string()
                        }),
                        _ => break
                    }
                },
                // parse access operation
                Some(Dot) => {
                    parsed_tokens.push(tokens.pop().unwrap());

                    match tokens.last() {
                        Some(Operator(_)) => continue,
                        _ => ()
                    }

                    let expr = parse! {
                        parsed_tokens; tokens, config;
                        parse_basic_expr();
                        parse_binary_expr(0);
                        parse_tail_expr()
                    };

                    result = AccessExpr {
                        from: Box::new(result),
                        to: Box::new(expr)
                    };
                    break
                },
                // parse link expression
                Some(Chain) => {
                    parsed_tokens.push(tokens.pop().unwrap());

                    let expr = parse! {
                        parsed_tokens; tokens, config;
                        parse_basic_expr();
                        parse_binary_expr(0);
                        parse_tail_expr()
                    };

                    // left-hand side expression type guard
                    if let Value(_) | ScopeExpr(_) = result {
                        let mut link = vec![result];
                        match expr {
                            LinkExpr(k) => link.extend(k),
                            _ => link.push(expr)
                        }

                        result = LinkExpr(link);
                        break
                    }

                    return Failure(InvalidExpressionError)
                },
                // parse assignment operation
                Some(Assign) => {
                    parsed_tokens.push(tokens.pop().unwrap());

                    let list = match result {
                        GroupExpr(group) => group,
                        _ => vec![result]
                    };

                    let expr = parse! {
                        parsed_tokens; tokens, config;
                        parse_primary_expr();
                        parse_binary_expr(0);
                        parse_tail_expr()
                    };

                    result = AssignExpr(list, Box::new(expr));
                    break
                },
                _ => break
            }
        );
    }

    loop {
        // continue until the current token is not an operator
        // or it is an operator with precedence lesser than expr_precedence
        let (operator, precedence) = match_operator! {
            Some(op) if op.precedence >= precedence => op.precedence
        };

        let token = tokens.pop().unwrap();
        parsed_tokens.push(token);

        // parse primary RHS expression
        let mut rhs = parse! {
            parsed_tokens;
            parse_primary_expr(tokens, config)
        };

        // parse all the RHS operators until their precedence is
        // bigger than the current one
        loop {
            let (_, binary_rhs) = match_operator! {
                Some(op) if op.precedence > precedence => parse! {
                    parsed_tokens;
                    parse_binary_expr(tokens, config, op.precedence, rhs)
                }
            };
            rhs = binary_rhs;
        }

        // merge LHS and RHS
        let call = CallExpr {
            name: operator,
            args: match rhs {
                GroupExpr(group) => group,
                expr => vec![expr]
            },
            generics: vec![]
        };

        result = AccessExpr {
            from: Box::new(result),
            to: Box::new(call)
        };
    }

    Success(result, parsed_tokens)
}

fn parse_tail_expr(tokens: &mut Vec<Token>, config: &Config, lhs: Expression) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();
    let mut result = lhs;

    if let Some(OpeningCurlyBrace) = tokens.last() {

        let expr = parse! {
            parsed_tokens;
            parse_primary_expr(tokens, config)
        };

        // tail block sugar - when followed by a block,
        // (1) a Value would turn into a CallExpr with the block as its only argument
        // (2) a CallExpr would take the block as its first argument
        result = match result {
            Value(name) => {
                CallExpr { name, args: vec![expr], generics: vec![] }
            },
            CallExpr { name, mut args, generics } => {
                args.reverse();
                args.push(expr);
                args.reverse();
                CallExpr { name, args, generics }
            },
            _ => result
        };
    }

    Success(result, parsed_tokens)
}

fn parse_basic_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();

    let scope = parse! {
        parsed_tokens;
        parse_scope_expr(tokens, config)
    };

    let name = match scope {
        Value(name) => name,
        _ => return Success(scope, parsed_tokens)
    };

    let expr = match tokens.last() {
        // as a function call
        // parse call arguments
        Some(OpeningParenthesis) => parse! {
            parsed_tokens;
            parse_primary_expr(tokens, config)
        },
        // function call with generics specified
        Some(Operator(x)) if x.eq("::<") => {
            parsed_tokens.push(tokens.pop().unwrap());

            let mut generics = Vec::new();
            enclosure!(generics, tokens, parsed_tokens, {
                (Operator(x) if x.eq(">")) => parse! {
                    parsed_tokens;
                    parse_type(tokens, config)
                },
                Comma => continue
            });

            match tokens.last() {
                Some(OpeningParenthesis) => parse! {
                    parsed_tokens;
                    parse_primary_expr(tokens, config)
                },
                _ => return Failure(InvalidExpressionError)
            }
        },
        // as a single value
        Some(_) | None => {
            let name = parse! {
                parsed_tokens;
                parse_annotated_value(tokens, config, name)
            };
            return Success(Value(name), parsed_tokens)
        }
    };

    let result = CallExpr {
        name,
        args: match expr {
            GroupExpr(list) => list,
            _ => vec![expr]
        },
        generics: vec![],
    };

    Success(result, parsed_tokens)
}

fn parse_annotated_value(tokens: &mut Vec<Token>, config: &Config, name: String) -> SnippetParsingResult<String> {
    let mut parsed_tokens = Vec::new();
    let mut result = name;

    if result.is_empty() {
        expect!(tokens, parsed_tokens, {
            Identifier(name) => result = name,
        });
    }

    // type annotation
    expect!(tokens, parsed_tokens, {
        Annotator => {
            let ty = parse! {
                parsed_tokens;
                parse_type(tokens, config)
            };

            // todo - type system
        },
        ? => ()
    });

    Success(result, parsed_tokens)
}

fn parse_scope_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();

    let result = expect!(tokens, parsed_tokens, {
        Identifier(x) if x.eq("self") => ScopeExpr("self".into()),
        By => ScopeExpr("by".into()),
        Identifier(name) => Value(name),
    });

    Success(result, parsed_tokens)
}

fn parse_number_literal_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();

    let expr = expect!(tokens, parsed_tokens, {
        Integer(val) => IntegerLiteralExpr(val),
        Float(val) => FloatLiteralExpr(val),
    });

    Success(expr, parsed_tokens)
}

fn parse_formatted_string_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();
    let mut format = Vec::new();

    let (texts, mats) =
        expect!(tokens, parsed_tokens, {
            FmtString(texts, mats) => (texts, mats),
        });

    for mut stack in mats {
        //
        if stack.is_empty() {
            continue
        }
        stack.reverse();

        let expr = parse! {
            Vec::<Token>::new();
            &mut stack, config;
            parse_primary_expr();
            parse_binary_expr(0)
        };

        format.push(expr);
    }

    let result = FmtStringExpr(texts, format);
    Success(result, parsed_tokens)
}

fn parse_string_literal_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();

    let text = expect!(tokens, parsed_tokens, {
        StringLiteral(val) => val,
    });

    Success(StringLiteralExpr(text), parsed_tokens)
}

fn parse_atom_expr(tokens: &mut Vec<Token>, config: &Config) -> ExpressionResult {
    let mut parsed_tokens = Vec::new();

    let val = expect!(tokens, parsed_tokens, {
        Atom(val) => val,
    });

    let expr = match val.to_lowercase().as_str() {
        "true" => BooleanExpr(true),
        "false" => BooleanExpr(false),
        _ => AtomExpr(val)
    };

    Success(expr, parsed_tokens)
}

fn parse_type(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<Type> {
    let mut parsed_tokens = Vec::new();

    let name = expect!(tokens, parsed_tokens, {
        Identifier(name) => name,
    });

    // todo - generics

    let result = Type {
        name,
        generics: vec![]
    };

    Success(result, parsed_tokens)
}

fn parse_generics(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<Vec<Type>> {
    let mut parsed_tokens = Vec::new();
    let mut generics = Vec::new();

    expect!(tokens, parsed_tokens, {
        Operator(x) if x.eq("<") =>
            enclosure!(generics, tokens, parsed_tokens, {
                (Operator(x) if x.eq(">")) => parse! {
                    parsed_tokens;
                    parse_type(tokens, config)
                },
                Comma => continue
            }),
        ? => ()
    });

    Success(generics, parsed_tokens)
}

fn parse_expression(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    let mut parsed_tokens = Vec::new();

    let expr = parse! {
        parsed_tokens; tokens, config;
        parse_primary_expr();
        parse_binary_expr(0);
        parse_tail_expr()
    };

    return Success(ASTNode::BasicNode(expr), parsed_tokens)
}

fn parse_decoration(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    let mut parsed_tokens = Vec::new();
    let mut decors = Vec::new();

    let node = loop {
        let token = tokens.pop().unwrap();
        parsed_tokens.push(token);

        if let Some(Decorator) = parsed_tokens.last() {
            let expr = parse! {
                parsed_tokens;
                parse_primary_expr(tokens, config)
            };

            decors.push(match expr {
                Value(_) | CallExpr { .. } => expr,
                _ => return Failure(InvalidExpressionError)
            });
        }

        match tokens.last() {
            Some(Decorator | Delimiter) => continue,

            Some(_) => break parse! {
                parsed_tokens;
                parse_node(tokens, config)
            },
            None => return Incomplete
        }
    };

    let result = ASTNode::DecorationNode {
        decorators: decors,
        node: Box::new(node)
    };

    Success(result, parsed_tokens)
}

fn parse_return(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    let mut parsed_tokens = Vec::new();
    parsed_tokens.push(tokens.pop().unwrap());

    let expr = match tokens.last() {
        Some(_) => parse! {
            parsed_tokens; tokens, config;
            parse_primary_expr();
            parse_binary_expr(0);
            parse_tail_expr()
        },
        None => AtomExpr("nil".into())
    };

    Success(ASTNode::ReturnNode(expr), parsed_tokens)
}

fn parse_module(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    let mut parsed_tokens = Vec::new();
    parsed_tokens.push(tokens.pop().unwrap());

    let name = expect!(tokens, parsed_tokens, {
        Identifier(name) => name,
    });

    // generics list
    let generics = parse! {
        parsed_tokens;
        parse_generics(tokens, config)
    };

    // todo - generics

    let includes = expect!(tokens, parsed_tokens, {
        By => {
            let mut result = Vec::new();
            enclosure!(result, tokens, parsed_tokens, {
                ((Delimiter | Do)) => parse! {
                    parsed_tokens;
                    parse_type(tokens, config)
                },
                Comma => continue
            });

            result
        },
        ? => vec![]
    });

    let mut body = Vec::new();
    enclosure!(body, tokens, parsed_tokens, {
        (End) => parse! {
            parsed_tokens;
            parse_node(tokens, config)
        },
        Delimiter => continue
    });

    let module = ast::Module {
        name,
        includes,
        body
    };

    Success(ASTNode::ModuleNode(module), parsed_tokens)
}

fn parse_prototype(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    let mut parsed_tokens = Vec::new();
    parsed_tokens.push(tokens.pop().unwrap());

    let is_trait =
        expect!(tokens, parsed_tokens, {
            Trait => true,
            ? => false
        });

    let name = expect!(tokens, parsed_tokens, {
        Identifier(name) => name,
    });

    let generics = parse! {
        parsed_tokens;
        parse_generics(tokens, config)
    };

    // todo - generics

    // function parameters
    let mut args = Vec::new();
    expect!(tokens, parsed_tokens, {
        OpeningParenthesis => (),
    });
    enclosure!(args, tokens, parsed_tokens, {
        (ClosingParenthesis) => parse! {
            parsed_tokens;
            parse_annotated_value(tokens, config, String::new())
        },
        Comma | Delimiter => continue
    });

    // return type
    expect!(tokens, parsed_tokens, {
        Arrow => {
            let ty = parse! {
                parsed_tokens;
                parse_type(tokens, config)
            };

            // todo - type system
        },
        ? => ()
    });

    let delegator =
        expect!(tokens, parsed_tokens, {
            By => Some(parse! {
                parsed_tokens;
                parse_type(tokens, config)
            }),
            ? => None
        });

    let prototype = Prototype {
        name,
        args,
        delegator,
    };

    let result = if is_trait {
        ASTNode::TraitNode(prototype)
    } else {
        // function body
        let mut body = Vec::new();
        expect!(tokens, parsed_tokens, {
            Do => {
                let expr = parse! {
                    parsed_tokens; tokens, config;
                    parse_primary_expr();
                    parse_binary_expr(0);
                    parse_tail_expr()
                };

                body.push(ASTNode::BasicNode(expr));
            },
            Delimiter => enclosure!(body, tokens, parsed_tokens, {
                (End) => parse! {
                    parsed_tokens;
                    parse_node(tokens, config)
                },
                Delimiter => continue
            }),
        });

        ASTNode::FunctionNode { prototype, body }
    };

    Success(result, parsed_tokens)
}

fn parse_begin_block(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    let mut parsed_tokens = Vec::new();
    parsed_tokens.push(tokens.pop().unwrap());

    // parameters
    let mut param = Vec::new();
    enclosure!(param, tokens, parsed_tokens, {
        (Delimiter) => parse! {
            parsed_tokens;
            parse_annotated_value(tokens, config, String::new())
        },
        Comma => continue
    });

    let mut body_begin = Vec::new();
    enclosure!(body_begin, tokens, parsed_tokens, {
        ((End | Final)) => parse! {
            parsed_tokens;
            parse_node(tokens, config)
        },
        Delimiter => continue
    });

    let mut body_final = Vec::new();
    if let Some(Final) = parsed_tokens.last() {
        enclosure!(body_final, tokens, parsed_tokens, {
            (End) => parse! {
                parsed_tokens;
                parse_node(tokens, config)
            },
            Delimiter => continue
        });
    }

    let result = ASTNode::BeginNode {
        param,
        body: (body_begin, body_final)
    };

    Success(result, parsed_tokens)
}

fn parse_import(tokens: &mut Vec<Token>, config: &Config) -> SnippetParsingResult<ASTNode> {
    let mut parsed_tokens = Vec::new();
    parsed_tokens.push(tokens.pop().unwrap());

    let only = expect!(tokens, parsed_tokens, {
        // import all
        Operator(x) if x.eq("*") => {
            expect!(tokens, parsed_tokens, {
                By => None,
            })
        },
        // import some
        ? => {
            let mut idents = Vec::new();
            enclosure!(idents, tokens, parsed_tokens, {
                (By) =>
                    expect!(tokens, parsed_tokens, {
                        Identifier(name) => name,
                    }),
                Comma => continue
            });
            Some(idents)
        }
    });

    let expr = parse! {
        parsed_tokens; tokens, config;
        parse_basic_expr();
        parse_binary_expr(0)
    };

    let from = match expr {
        LinkExpr(link) => {
            let mut path = Vec::new();
            for bit in link {
                if let Value(name) = bit {
                    path.push(name);
                    continue
                }
                return Failure(InvalidExpressionError)
            }
            path
        },
        Value(name) => vec![name],
        _ => return Failure(InvalidExpressionError)
    };

    let result = ASTNode::ImportNode {
        only, from
    };

    return Success(result, parsed_tokens)
}
