#[derive(Debug, Clone)]
pub enum ASTNode {
    BasicNode(Expression),
    ModuleNode(Module),
    DecorationNode {
        decorators: Vec<Expression>,
        node: Box<ASTNode>
    },
    FunctionNode {
        prototype: Prototype,
        body: Vec<ASTNode>
    },
    TraitNode(Prototype),
    ImportNode {
        only: Option<Vec<String>>,
        from: Vec<String>
    },
    BeginNode {
        param: Vec<String>,
        body: (Vec<ASTNode>, Vec<ASTNode>)
    },
    ReturnNode(Expression)
}

#[derive(Debug, Clone)]
pub enum Expression {
    AtomExpr(String),
    BooleanExpr(bool),
    StringLiteralExpr(String),
    FmtStringExpr(Vec<String>, Vec<Expression>),
    IntegerLiteralExpr(i32),
    FloatLiteralExpr(f64),

    ConditionalExpr {
        cond_expr: Box<Expression>,
        then_expr: Box<Expression>,
        else_expr: Box<Expression>
    },
    LoopExpr {
        item: String,
        iterable: Box<Expression>,
        step: Box<Expression>,
        body: Box<Expression>
    },

    ScopeExpr(String),
    Value(String),

    CallExpr {
        name: String,
        args: Vec<Expression>,
        generics: Vec<Type>
    },
    AssignExpr(Vec<Expression>, Box<Expression>),
    AccessExpr {
        from: Box<Expression>,
        to: Box<Expression>
    },
    LinkExpr(Vec<Expression>),

    BlockExpr {
        args: Vec<String>,
        body: Vec<ASTNode>
    },
    GroupExpr(Vec<Expression>),
    ListExpr(Vec<Expression>)
}

#[derive(Debug, Clone)]
pub struct Type {
    pub(crate) name: String,
    pub(crate) generics: Vec<Type>
}

#[derive(Debug, Clone)]
pub struct Module {
    pub(crate) name: String,
    pub(crate) includes: Vec<Type>,
    pub(crate) body: Vec<ASTNode>
}

#[derive(Debug, Clone)]
pub struct Prototype {
    pub(crate) name: String,
    pub(crate) args: Vec<String>,
    pub(crate) delegator: Option<Type>
}
