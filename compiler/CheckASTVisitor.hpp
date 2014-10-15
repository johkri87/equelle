/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.
*/

#ifndef OPM_CHECKASTVISITOR_HEADER_INCLUDED
#define OPM_CHECKASTVISITOR_HEADER_INCLUDED


#include "ASTVisitorInterface.hpp"
#include "FileLocation.hpp"
#include <string>
#include <stack>


class CheckASTVisitor : public ASTVisitorInterface
{
public:
    CheckASTVisitor();
    virtual ~CheckASTVisitor();

    void visit(SequenceNode& node);
    void midVisit(SequenceNode& node);
    void postVisit(SequenceNode& node);
    void visit(NumberNode& node);
    void visit(QuantityNode& node);
    void postVisit(QuantityNode& node);
    void postVisit(BinaryOpUnitNode& node);
    void visit(StringNode& node);
    void visit(TypeNode& node);
    void visit(CollectionTypeNode& node);
    void postVisit(CollectionTypeNode& node);
    void visit(FuncTypeNode& node);
    void visit(BinaryOpNode& node);
    void midVisit(BinaryOpNode& node);
    void postVisit(BinaryOpNode& node);
    void visit(ComparisonOpNode& node);
    void midVisit(ComparisonOpNode& node);
    void postVisit(ComparisonOpNode& node);
    void visit(NormNode& node);
    void postVisit(NormNode& node);
    void visit(UnaryNegationNode& node);
    void postVisit(UnaryNegationNode& node);
    void visit(OnNode& node);
    void midVisit(OnNode& node);
    void postVisit(OnNode& node);
    void visit(TrinaryIfNode& node);
    void questionMarkVisit(TrinaryIfNode& node);
    void colonVisit(TrinaryIfNode& node);
    void postVisit(TrinaryIfNode& node);
    void visit(VarDeclNode& node);
    void postVisit(VarDeclNode& node);
    void visit(VarAssignNode& node);
    void postVisit(VarAssignNode& node);
    void visit(VarNode& node);
    void visit(FuncRefNode& node);
    void visit(JustAnIdentifierNode& node);
    void visit(FuncArgsDeclNode& node);
    void midVisit(FuncArgsDeclNode& node);
    void postVisit(FuncArgsDeclNode& node);
    void visit(FuncDeclNode& node);
    void postVisit(FuncDeclNode& node);
    void visit(FuncStartNode& node);
    void postVisit(FuncStartNode& node);
    void visit(FuncAssignNode& node);
    void postVisit(FuncAssignNode& node);
    void visit(FuncArgsNode& node);
    void midVisit(FuncArgsNode& node);
    void postVisit(FuncArgsNode& node);
    void visit(ReturnStatementNode& node);
    void postVisit(ReturnStatementNode& node);
    void visit(FuncCallNode& node);
    void postVisit(FuncCallNode& node);
    void visit(FuncCallStatementNode& node);
    void postVisit(FuncCallStatementNode& node);
    void visit(LoopNode& node);
    void postVisit(LoopNode& node);
    void visit(ArrayNode& node);
    void postVisit(ArrayNode& node);
    void visit(RandomAccessNode& node);
    void postVisit(RandomAccessNode& node);
    void visit(StencilAssignmentNode& node);
    void midVisit(StencilAssignmentNode& node);
    void postVisit(StencilAssignmentNode& node);
    void visit(StencilNode& node);
    void postVisit(StencilNode& node);

private:
    int checking_suppression_level_;
    int next_loop_index_;
    std::stack<std::string> undecl_func_stack;

    void error(const std::string& err, const FileLocation loc = FileLocation());
    // Note that the suppression works like a stack:
    // each suppress call increases the supression level
    // and each unsuppress call decreases it, so checking may
    // not be unsuppressed by as single unsupress call, if more
    // than one suppress call preceded it.
    void suppressChecking();
    void unsuppressChecking();
    bool isCheckingSuppressed() const;
};


#endif // OPM_CHECKASTVISITOR_HEADER_INCLUDED
