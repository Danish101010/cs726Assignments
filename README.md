# Assignment1
# Junction Tree Inference in Graphical Models

This project implements **exact inference** in probabilistic graphical models using the **Junction Tree Algorithm**. The full pipeline includes graph triangulation, junction tree construction, message passing, and marginal computation.

## 🚀 Features

- **Graph Triangulation**: 
  - Detects simplicial vertices
  - Adds fill-in edges to make the graph chordal

- **Junction Tree Construction**: 
  - Builds the tree from cliques using a minimum spanning tree of the clique graph

- **Potential Assignment**: 
  - Assigns potential functions to cliques with proper indexing

- **Message Passing**: 
  - Performs sum-product message passing
  - Computes clique and variable marginals
  - Computes the partition function (Z)

- **Top-K Assignments**: 
  - Extracts the most probable variable assignments from the model

## 📁 Structure

- `triangulation.py`: Chordal graph construction
- `junction_tree.py`: Tree building logic
- `potentials.py`: Potential indexing and initialization
- `message_passing.py`: Belief propagation logic
- `top_k.py`: Computes top-K probable assignments

## 📚 Conclusion

The implementation offers a complete toolkit for inference in graphical models, useful for applications like probabilistic reasoning, computer vision, and bioinformatics.

---

Feel free to clone, test, and extend the work!



