use std::collections::HashMap;

macro_rules! config_operator {
    ($($($operator:literal)|+ => $precedence:literal),+) => {{
        let mut operators = HashMap::new();
        $($(
            let name = $operator.to_string();
            operators.insert(name.clone(), Operator {
                name,
                precedence: $precedence
            });
        )+)+
        operators
    }};
}

pub struct ParserConfiguration {
    pub operators: HashMap<String, Operator>,
}

pub fn default_parser_config() -> ParserConfiguration {
    let operators = config_operator! {
        "=" => 2,
        "<" | ">" => 10,
        "+" | "-" => 20,
        "*" | "/" => 40
    };

    ParserConfiguration {
        operators,
    }
}

pub struct Operator {
    pub name: String,
    pub precedence: i32
}
