def rewrite_graph(graph):
    for node in graph.node:
        if node.name == "Log":
            node.name = "TestLog"
        if node.name == "gradients/mul_grad/Shape_1":
            node.input[0] = "TestLog"
        if node.name == "gradients/mul_grad/Mul":
            node.input[1] = "TestLog"
        if node.name == "mul":
            node.input[1] = "TestLog"

def rewrite_stepstats(step_stats):
    for dev_stat in step_stats.dev_stats:
        for node_stat in dev_stat.node_stats:
            if node_stat.node_name == "Log":
                node_stat.node_name = "TestLog"

def rewrite_meta(run_metadata):
    rewrite_graph(run_metadata.partition_graphs[0])
    rewrite_stepstats(run_metadata.step_stats)
