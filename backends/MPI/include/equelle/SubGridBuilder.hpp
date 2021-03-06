#pragma once

#include <vector>
#include <unordered_map>
#include <set>

#include "equelle/equelleTypes.hpp"

class UnstructuredGrid;


namespace equelle {

/** UnstructuredSubGrid is as an augmented Opm::UnstructuredGrid */
struct SubGrid {
    UnstructuredGrid *c_grid; //! Pointer to data for the subGrid with "local"-indexing. Owned by this class.

    int number_of_ghost_cells;

    std::vector<int> cell_local_to_global; //! Maps local cell indices to global cell indices. The ghost cells are the
                                  //! last cells in this range.

    std::unordered_map<int, int> cell_global_to_local; //! Maps global cell indices to local cell indices.
                                                       //! This is the inverse of global_cell

    std::vector<int> face_local_to_global; //! Maps local face indices to global face indices.
    std::unordered_map<int, int> face_global_to_local; //! Maps global face indices to local face indices.
                                                       //! This is the inverse of global_face.
    CollOfCell map_to_global( const CollOfCell& local_collection );
    CollOfFace map_to_global( const CollOfFace& local_collection );

    ~SubGrid();
};

/** SubGridBuilder is responsible for building a subgrid of an Opm::UnstructuredGrid,
 *  given the export list from Zoltan.
 */
class SubGridBuilder
{
public:
    /**
     * @brief build a SubGrid from a list of global cell indices.
     * @param globalGrid
     * @param cellsToExtract Global cell IDs of cells that shall be owned by the SubGrid.
     *        In addition the returned subGrid will contain a number of ghost cells.
     *        Need not be in sorted order.
     *        The first (SubGrid.c_grid->number_of_cells - SubGrid::number_of_ghost_cells) elements of
     *        SubGrid::global_cell will be equal to this parameter.
     * @return A new SubGrid
     */
    static SubGrid build( const UnstructuredGrid* globalGrid, const std::vector<int>& cellsToExtract );


private:
    SubGridBuilder();

    /**
     * @brief The face_mapping struct is a helper return type in the implemenation of build.
     */

    struct face_mapping {
        std::vector<int> cell_facepos; //! Mirrors UnstructuredGrid::cell_facepos.
        std::vector<int> cell_faces;   //! Mirrors UnstructuredGrid::cell_faces.
        std::vector<int> global_face;  //! The global face index of each face in the subgrid.
        std::unordered_map<int, int> face_global_to_local; //! The inverse of global_face.
    };

    /**
     * @brief The node_mapping struct is a helper return type in the implementation of build.
     */
    struct node_mapping {
        std::vector<int> face_nodepos; //! Mirrors UnstructuredGrid::face_nodepos;
        std::vector<int> face_nodes;   //! Mirrors UnstructuredGrid::face_nodes;
        std::vector<int> global_node;  //! The global node index of each node in the subgrid.
        std::unordered_map<int, int> node_global_to_local; //! The inverse of global_node.
    };

    static std::set<int> extractNeighborCells(const UnstructuredGrid *grid, const std::vector<int>& cellsToExtract);
    static face_mapping extractNeighborFaces(const UnstructuredGrid *grid, const std::vector<int>& cellsToExtract);
    static node_mapping extractNeighborNodes(const UnstructuredGrid *grid, const std::vector<int>& globalFaces);

    static void build_face_cells( const face_mapping& participatingFaces, SubGrid& subGrid, const UnstructuredGrid* grid );
};

struct GridQuerying {
    /** Return the number of faces for a cell. */
    static int numFaces( const UnstructuredGrid* grid, int cell );

    /** Return the number of nodes for a face. */
    static int numNodes( const UnstructuredGrid* grid, int face );        
};

} // namespace equelle



