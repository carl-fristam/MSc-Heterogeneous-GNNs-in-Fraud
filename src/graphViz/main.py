"""
Phase 1: Build Basic Graph Representation from HI-Small AML Dataset

Main entry point for graph construction pipeline.
"""

from graph import (
    load_data,
    create_account_mapping,
    build_transaction_graph,
    add_account_attributes,
    compute_all_statistics,
    print_statistics,
    visualize_subgraph,
    create_degree_distribution_plot,
    save_graph
)


def main():
    """Execute the graph building pipeline."""
    print("="*60)
    print("PHASE 1: BASIC GRAPH REPRESENTATION")
    print("="*60)
    
    # Step 1: Load data
    trans_df, accounts_df = load_data()
    
    # Step 2: Create account mapping
    account_to_id, all_accounts = create_account_mapping(trans_df, accounts_df)
    
    # Step 3: Build graph
    G = build_transaction_graph(trans_df, account_to_id)
    
    # Step 4: Add account attributes
    G = add_account_attributes(G, accounts_df, account_to_id)
    
    # Step 5: Compute statistics
    stats = compute_all_statistics(G)
    print_statistics(stats)
    
    # Step 6: Visualize
    visualize_subgraph(G)
    create_degree_distribution_plot(G)
    
    # Step 7: Save
    save_graph(G, account_to_id)
    
    print("\n" + "="*60)
    print("COMPLETE! Graph saved to outputs/")
    print("="*60)
    print("\nNext steps:")
    print("  1. Explore the graph: outputs/transaction_graph.gpickle")
    print("  2. View visualization: outputs/subgraph.png")
    print("  3. View degree distribution: outputs/degree_distribution.png")
    print("  4. Open in Gephi/Cytoscape: outputs/transaction_graph.graphml")
    print("  5. Ready for Phase 2: GraphSAGE implementation")


if __name__ == "__main__":
    main()
