/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PRINTCPUBACKENDASTVISITOR_HEADER_INCLUDED
#define PRINTCPUBACKENDASTVISITOR_HEADER_INCLUDED

#include "ASTVisitorInterface.hpp"
#include "EquelleType.hpp"
#include <string>
#include <set>

class PrintCPUBackendASTVisitor : public ASTVisitorInterface
{
public:
    PrintCPUBackendASTVisitor();
    PrintCPUBackendASTVisitor(const bool use_cartesian);
    virtual ~PrintCPUBackendASTVisitor();

    void visit(SequenceNode& node);
    void midVisit(SequenceNode& node);
    void postVisit(SequenceNode& node);
    void visit(NumberNode& node);
    void visit(QuantityNode& node);
    void postVisit(QuantityNode& node);
    void visit(StringNode& node);
    void visit(TypeNode& node);
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

    // These are overriden by subclasses who only need to alter the surroundings of the generated code.
    virtual const char* cppStartString() const;
    virtual const char* cppEndString() const;
    virtual const char* classNameString() const;
    virtual const char* namespaceNameString() const;

private:
    int suppression_level_;
    int indent_;
    int sequence_depth_;
    std::set<std::string> requirement_strings_;
    std::set<std::string> defined_mutables_;
    bool instantiating_;
    int next_funcstart_inst_;
    std::string skipping_function_;
    bool use_cartesian_;

    void endl() const;
    std::string indent() const;
    void suppress();
    void unsuppress();
    bool isSuppressed() const;
    std::string cppTypeString(const EquelleType& et) const;
    void addRequirementString(const std::string& req);
};

#endif // PRINTCPUBACKENDASTVISITOR_HEADER_INCLUDED
