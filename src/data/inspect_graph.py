import argparse
import pickle
import sys

from src.utils.config import PROJECT_ROOT


def inspect(cache_path):
    """Load and inspect a cached graph."""
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {cache_path} not found")
        sys.exit(1)

    data = cache['data']
    account_mapping = cache['account_to_id']

    print("=" * 60)
    print("GRAPH DATA INSPECTION")
    print("=" * 60)
    print(f"\nData object: {data}")
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Node features: {data.num_node_features}")

    print(f"\nLabel Distribution:")
    print(f"  Clean accounts (0): {(data.y == 0).sum().item()}")
    print(f"  Laundering accounts (1): {(data.y == 1).sum().item()}")
    print(f"  Positive rate: {(data.y == 1).sum().item() / len(data.y) * 100:.2f}%")

    print(f"\nFirst 5 nodes' features:")
    print(data.x[:5])

    print(f"\nFirst 10 edges (source -> target | edge_label):")
    edges = data.edge_index[:, :10].T
    for i in range(min(10, edges.shape[0])):
        src, tgt = edges[i]
        edge_label = data.edge_attr[i].item()
        print(f"  Edge {i}: Node {src.item()} -> Node {tgt.item()} | Label: {int(edge_label)}")

    print(f"\nSample account mappings (first 5):")
    for i, (acc_id, node_idx) in enumerate(list(account_mapping.items())[:5]):
        node_label = data.y[node_idx].item()
        print(f"  {i+1}. Account {acc_id} -> Node {node_idx} | Label: {int(node_label)}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect a cached PyG graph")
    parser.add_argument(
        'cache_path',
        nargs='?',
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'saml_graph.pkl'),
        help='Path to the cached .pkl graph file',
    )
    args = parser.parse_args()
    inspect(args.cache_path)
