// deal.ii library directives
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/logstream.h>

// cpp library directives
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>

using namespace dealii;

// Class Definitions
template <int dim>
class pcbSteadyState
{
    public:
      pcbSteadyState();
      void run();
      void make_grid_from_vtk();      
    private:
      void make_grid();
      void setup_system();
      void assemble_system();
      void solve();
      void output_results() const;
      Triangulation<dim> triangulation;
      FE_Q<dim>          fe;
      DoFHandler<dim>    dof_handler;
      SparsityPattern      sparsity_pattern;
      SparseMatrix<double> system_matrix;
      Vector<double> solution;
      Vector<double> system_rhs;
};

template <int dim>
class HeatSource : public Function<dim>    
// Nice example of inheritance. 'HeatSource' inherits from 'Function'
{
    public:
      virtual double value(const Point<dim> & p,  // the 'virtual' keyword tells cpp that 
                                                  // this function can be overwritten if it
                                                  // is redefined verbosely
                          const unsigned int component = 0) const override;
};

template <int dim>
class BoundaryValues : public Function<dim>
{
    public:
      virtual double value(const Point<dim> & p,
                          const unsigned int component = 0) const override;
};

// Methods Definitions
template <int dim>
double HeatSource<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
      double return_value = 4.287e9;        // This is the source
      return return_value;
}

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
    // This function is not necessarily critical since I am simply setting 
    // the boundary temperature to 0
    // I will omit this in later implementations because for our problem class
    // we do not enforce any temperature 
    return 0;
}

// Default Constructor - sets up the finite element object, fe and the mesh  
// object called triangulation
template <int dim>
pcbSteadyState<dim>::pcbSteadyState()
  : fe(1)
  , dof_handler(triangulation)
{}

template <int dim>
void pcbSteadyState<dim>::make_grid()
{
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(4);
    // Here we are setting two of the faces to have a boundary ID of 1 based on
    // where the centers of the two faces are located. 
    // In practice I want to read the boundary values to in the VTK file, so this 
    // loop will change significantly  
    // This approach could actually be useful because it can allow me to set the
    // top and bottom faces boundary conditions without knowing the face IDs
    
    // for (const auto &cell : triangulation.cell_iterators())
    //   for (const auto &face : cell->face_iterators())
    //     {
    //       const auto center = face->center();
    //       if ((std::fabs(center(0) - (-1.0)) < 1e-12) ||
    //           (std::fabs(center(1) - (-1.0)) < 1e-12))
    //         face->set_boundary_id(1);
    //     }

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Total number of cells: " << triangulation.n_cells()
              << std::endl;
}

template <int dim>
void pcbSteadyState<dim>::make_grid_from_vtk()
{
  GridIn<dim, dim> pcb_input_mesh;
  pcb_input_mesh.attach_triangulation(triangulation);
  std::ifstream vtk_input("testPCBMesh.vtk");      //let's read in the vtk input file
  pcb_input_mesh.read_vtk(vtk_input);
  //Let's print out some mesh info to ensure the code worked
  std::cout <<"Viola! Look at that - we have just read the mesh successfully.\nMesh info:" << std::endl
            << "\tDimension: " << 3 << std::endl
            << "\tno. of cells: " << triangulation.n_active_cells() << std::endl;
  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void pcbSteadyState<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void pcbSteadyState<dim>::assemble_system()
{
    QGauss<dim>           quadrature_formula(fe.degree + 1);
    QGauss<dim - 1>       face_quadrature_formula(fe.degree + 1); // added to loop through defined faces
    const unsigned int n_q_points      = quadrature_formula.size(); // number of quadrature points
    const unsigned int n_face_q_points = face_quadrature_formula.size(); // number of faces on the Neumann boundary
    HeatSource<dim>  heat_gen;
    int curr_cell = 0;
    FEValues<dim>         fe_values(fe,
                                  quadrature_formula,
                                  update_values | update_gradients |  
                                  update_quadrature_points | update_JxW_values);
    FEFaceValues<dim>     fe_face_values(fe,                    // added from step 7
                                  face_quadrature_formula,
                                  update_values | update_quadrature_points |
                                  update_normal_vectors |
                                  update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FullMatrix<double>    cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>        cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Read Material Properties (all hardcoded for now)
    int num_cells = triangulation.n_cells();
    int idx_vector[num_cells];
    for (int i = 0; i < num_cells; i++){
      idx_vector[i] = i;
    }
    
    // Read material IDs and their respective properties
    std::ifstream material_ids("material_id.txt");
    std::istream_iterator<int> start(material_ids), end;
    std::vector<int> geomDef(start, end);

    // Read heat source cells
    std::ifstream heat_source_idx("heat_sources.txt");
    std::istream_iterator<int> first(heat_source_idx), last;
    std::vector<int> heatDef(first, last);

    // end of material property set up
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
              // Here I think I need to track the direction of the shape function (in x,y,z)
              // based on the element of the 8x8 matrix defining the local conduction
              // matrix. From this knowledge, I can then apply a simple switch-case conditional
              // to assign the directional thermal conductivity
              // Also needed - a matrix with the material properties that can be used to assign 
              // the correct thermal conductivity based on the identity of the cell

              if(geomDef[curr_cell] == 1) // check if the current cell is copper and assign
                                          // thermal_cond to 400 if so
              {
                    cell_matrix(i, j) +=
                                  (400*fe_values.shape_grad(i, q_index) *   // grad phi_i(x_q)
                                  fe_values.shape_grad(j, q_index) *        // grad phi_j(x_q)
                                  fe_values.JxW(q_index));                  // dx
              } else {                    // otherwise assign thermal_cond to be 0.3
                    cell_matrix(i, j) +=
                                  (0.3*fe_values.shape_grad(i, q_index) *   // grad phi_i(x_q)
                                  fe_values.shape_grad(j, q_index) *        // grad phi_j(x_q)
                                  fe_values.JxW(q_index));
              }                
              const auto &x_q = fe_values.quadrature_point(q_index);
              // std::cout << "We are on cell: " << curr_cell << "\n";
              
              // Now let's check if the cell we are on is a heat source cell
              auto itr = find(heatDef.begin(), heatDef.end(), curr_cell);
              if(itr != heatDef.end())                                // this is the return value if the find operation fails
              {
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *   // phi_i(x_q)
                                heat_gen.value(x_q) *                 // f(x_q)
                                fe_values.JxW(q_index));              // dx
              } else {
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *   // phi_i(x_q)
                                0 *                                   // f(x_q)
                                fe_values.JxW(q_index));              // dx
              }
            }
        // // Here is where we apply the Neumann Boundary Conditions (Reserved for the Heat Source)   
        // for (const auto &face : cell->face_iterators())
        //   if (face->at_boundary() && (face->boundary_id() == 1))
        //     {
        //       fe_face_values.reinit(cell, face);
        //       for (unsigned int q_point = 0; q_point < n_face_q_points;
        //            ++q_point)
        //         {
        //           const double heat_flux_bc = 10;
        //           for (unsigned int i = 0; i < dofs_per_cell; ++i)
        //             cell_rhs(i) +=
        //               (fe_face_values.shape_value(i, q_point) *     // N(i) - the value of the shape function at the GQ point
        //                heat_flux_bc *                               // heat flux BC in W/m^2. Note that we aren't using h, the convective coefficient
        //                fe_face_values.JxW(q_point));                // dx
        //         }
        //     }

        // Here let's apply the Robin Boundary Conditions (aka convective heat flux)
        for (unsigned int face = 0; face <GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary())
            {
                fe_face_values.reinit (cell, face);
                for (unsigned int i=0; i < dofs_per_cell; ++i)
                    for (unsigned int j=0; j < dofs_per_cell; ++j)
                        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                        {
                          cell_matrix(i, j) += (10 * fe_face_values.shape_value(i, q_point) *
                                                    fe_face_values.shape_value(j, q_point)) *
                                                    fe_face_values.JxW(q_point);
                        }
            }

        // Assemble the calculated stiffness matrix entries into global K and F
        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      curr_cell += 1;
      }
    std::map<types::global_dof_index, double> boundary_values;

    // Commenting out the constant temperature boundary condition below
    // VectorTools::interpolate_boundary_values(dof_handler,
    //                                         0,
    //                                         BoundaryValues<dim>(),
    //                                         boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}

template <int dim>
void pcbSteadyState<dim>::solve()
{
    SolverControl            solver_control(90000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;
}
template <int dim>
void pcbSteadyState<dim>::output_results() const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
    data_out.write_vtk(output);
}

template <int dim>
void pcbSteadyState<dim>::run()
{
    std::cout << "Solving problem in " << dim << " space dimensions."
              << std::endl;
    // make_grid();
    make_grid_from_vtk();
    setup_system();
    assemble_system();
    solve();
    output_results();
}

// Implementation in main
int main()
{
  {
    pcbSteadyState<3> boring_pcb;
    boring_pcb.run();
  }
  return 0;
}