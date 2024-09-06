import numpy as np
import argparse
from pathlib import Path


def compute_latency(timestamp0: str, timestamp1: str) -> float:
    return float(int(timestamp1) - int(timestamp0)) / 1000.0  # convert to ms


def get_node_latency(op_timestamps: list, node_latencies: dict) -> None:
    for op_timestamp in op_timestamps:
        op_name = op_timestamp[0]
        latency = compute_latency(op_timestamp[1], op_timestamp[2])  # ms
        if op_name not in node_latencies:
            node_latencies[op_name] = [latency]
        else:
            # Sometimes for some reason to be investigated there are duplicate latencies; this is also the same for the example code in the
            # Holohub repository: https://github.com/nvidia-holoscan/holohub/tree/main/benchmarks/holoscan_flow_benchmarking
            last_latency = node_latencies[op_name][-1]
            if latency == last_latency:
                continue
            node_latencies[op_name].append(latency)


def get_edge_latency(op_timestamps: list, edge_latencies: dict) -> None:
    num_ops = len(op_timestamps)
    for i in range(num_ops - 1):
        source_op = op_timestamps[i]
        sink_op = op_timestamps[i + 1]
        edge_name = f"{source_op[0]} --> {sink_op[0]}"  # mermaid format
        latency = compute_latency(source_op[2], sink_op[1])  # ms
        if edge_name not in edge_latencies:
            edge_latencies[edge_name] = [latency]
        else:
            last_latency = edge_latencies[edge_name][-1]
            if latency == last_latency:
                continue
            edge_latencies[edge_name].append(latency)


def get_path_latency(op_timestamps: list, path_latencies: dict) -> None:
    first_op = op_timestamps[0]
    last_op = op_timestamps[-1]
    path_name = f"{first_op[0]} --> {last_op[0]}"  # mermaid format
    latency = compute_latency(first_op[1], last_op[2])  # ms
    if path_name not in path_latencies:
        path_latencies[path_name] = [latency]
    else:
        last_latency = path_latencies[path_name][-1]
        if last_latency == latency:
            return
        path_latencies[path_name].append(latency)


def parse_line_from_log(line: str) -> list:
    operators = line.split("->")
    op_timestamps = []
    for operator in operators:
        # trim whitespaces for left and right side
        op_name_timestamp = operator.strip().rstrip()[1:-1]
        op_timestamps.append(op_name_timestamp.split(","))
    return op_timestamps


def main(log_path: Path, num_start_messages_to_skip: int) -> None:
    r"""
    Analyze the node, edge, and path latencies from the log file, creates a json file with the statistics.
    Args:
        log_path (Path): Path to the log file
        num_start_messages_to_skip (int): Number of start messages to skip
    """

    node_latencies = (
        {}
    )  # For each node in the graph; time spent in the computation of the node
    edge_latencies = (
        {}
    )  # For each edge in the graph; time spent in the communication between nodes
    path_latencies = (
        {}
    )  # For each line from the beginning to the end of the forward path; time spent in the computation and communication of the whole path

    num_lines = 0
    with open(log_path, "r") as f:
        for line in f:
            if line[0] == "(":
                if num_start_messages_to_skip <= num_lines:
                    op_timestamps = parse_line_from_log(line)
                    get_node_latency(op_timestamps, node_latencies)
                    get_edge_latency(op_timestamps, edge_latencies)
                    get_path_latency(op_timestamps, path_latencies)
                num_lines += 1

    node_latencies_stats = {}
    node_latencies_stats["description"] = "Intra-node (operator) computation latency statistics for each node in the graph; values are in ms"
    for node, latencies in node_latencies.items():
        node_latency_stats = {}
        latencies = np.array(latencies)
        node_latency_stats["median"] = np.median(latencies)
        node_latency_stats["mean"] = np.mean(latencies)
        node_latency_stats["std"] = np.std(latencies)
        node_latency_stats["min"] = np.min(latencies)
        node_latency_stats["max"] = np.max(latencies)
        node_latency_stats["99th percentile"] = np.percentile(latencies, 99)
        node_latency_stats["95th percentile"] = np.percentile(latencies, 95)
        node_latencies_stats[node] = node_latency_stats

    edge_latencies_stats = {}
    edge_latencies_stats["description"] = "Inter-node (operator) communication latency statistics for each edge in the graph; values are in ms"
    for edge, latencies in edge_latencies.items():
        edge_latency_stats = {}
        latencies = np.array(latencies)
        edge_latency_stats["median"] = np.median(latencies)
        edge_latency_stats["mean"] = np.mean(latencies)
        edge_latency_stats["std"] = np.std(latencies)
        edge_latency_stats["min"] = np.min(latencies)
        edge_latency_stats["max"] = np.max(latencies)
        edge_latency_stats["99th percentile"] = np.percentile(latencies, 99)
        edge_latency_stats["95th percentile"] = np.percentile(latencies, 95)
        edge_latencies_stats[edge] = edge_latency_stats

    path_latencies_stats = {}
    path_latencies_stats["description"] = "Computation and communication latency along a full forward path in the graph; this is the sum of the latencies of all nodes (operators) and edges in the path; values are in ms"
    for path, latencies in path_latencies.items():
        path_latency_stats = {}
        latencies = np.array(latencies)
        path_latency_stats["median"] = np.median(latencies)
        path_latency_stats["mean"] = np.mean(latencies)
        path_latency_stats["std"] = np.std(latencies)
        path_latency_stats["min"] = np.min(latencies)
        path_latency_stats["max"] = np.max(latencies)
        path_latency_stats["99th percentile"] = np.percentile(latencies, 99)
        path_latency_stats["95th percentile"] = np.percentile(latencies, 95)
        path_latencies_stats[path] = path_latency_stats
    
    # Dump a json file wit the stats
    import json
    with open(log_path.with_suffix(".stats.json"), "w") as f:
        json.dump(
            {
                "node_latencies": node_latencies_stats,
                "edge_latencies": edge_latencies_stats,
                "path_latencies": path_latencies_stats,
            },
            f,
            indent=4,
        )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Analyze the latencies in the log files. Usage: python log_analyzer.py -l <log-path>")
    argparser.add_argument(
        "-l",
        "--log-path",
        type=Path,
        help="Path to the log file",
        required=True,
    )
    argparser.add_argument(
        "-n",
        "--num-start-messages-to-skip",
        type=int,
        help="Number of start messages to skip",
        default=20
    )


    # Usage help string
    args = argparser.parse_args()
    main(args.log_path, args.num_start_messages_to_skip)