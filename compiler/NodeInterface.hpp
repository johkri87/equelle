/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef NODEINTERFACE_HEADER_INCLUDED
#define NODEINTERFACE_HEADER_INCLUDED

#include "EquelleType.hpp"
#include "FileLocation.hpp"

class ASTVisitorInterface;

/// Base class for all AST classes.
class Node
{
public:
    Node()
    {}
    virtual ~Node()
    {}
    virtual void accept(ASTVisitorInterface&)
    {
        // Do nothing.
    }
    FileLocation location() const
    {
        return loc_;
    }
    void setLocation(const FileLocation& loc)
    {
        loc_ = loc;
    }

    /*virtual int numChildren() = 0;
    virtual Node* getChild(const int index) = 0;
    virtual void setChild(const int index, Node* child) = 0;
    virtual Node* getParent() = 0;*/
    virtual void rewrite()
    {
        
    }
private:
    // No copying.
    Node(const Node&);
    // No assignment.
    Node& operator=(const Node&);
    FileLocation loc_;
};


#endif // NODEINTERFACE_HEADER_INCLUDED
