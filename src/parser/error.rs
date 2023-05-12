use custom_error::custom_error;

custom_error! { pub LexError
    UnrecognizedTokenError { token: String } = "Unrecognized Token {token}",
    InvalidNumberError { token: String } = "Lexer failed at parsing number {token}",
}

custom_error! { pub ParseError
    UnexpectedTokenError { token: String } = "Unexpected token '{token}'",
    InvalidArgumentListError = "",
    InvalidLiteralError = "Token provided is not a valid literal",
    MissingExpectedTokenError { tokens: Vec<String> } = @{{
        let mut message = "".to_string();
        let mut tokens = tokens.clone();

        loop { message = match tokens.len() {
            0 => break format!("{} expected.", message.trim_start()),
            1 => format!(" '{}'", tokens.pop().unwrap()),
            2 => format!(" '{}' or '{}'", tokens.pop().unwrap(), tokens.pop().unwrap()),
            _ => format!(" '{}',", tokens.pop().unwrap())
        }}
    }},
    UndefinedOperatorError { name: String } = "Unknown operator '{name}'",
    InvalidExpressionError = "Invalid Expression"
}
