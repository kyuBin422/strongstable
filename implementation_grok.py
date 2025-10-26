import copy
import networkx as nx
from collections import defaultdict, deque


def get_f(prefs, p):
    """
    Get the head (most preferred tied agents) of p's preference list.
    """
    return prefs[p][0] if prefs[p] else []


def get_l(prefs, p):
    """
    Get the tail (least preferred tied agents) of p's preference list.
    """
    return prefs[p][-1] if prefs[p] else []


def get_rank(prefs, agent, target):
    """
    Get the rank of target in agent's preference list.
    Lower rank is better.
    """
    for i, group in enumerate(prefs[agent]):
        if target in group:
            return i
    raise ValueError(f"Target {target} not found in {agent}'s preference list")


def delete_pair(prefs, a, b, semi=None, semi_to=None):
    """
    Delete the pair {a, b} from preference lists and break semi-assignments if exist.
    """
    # Remove b from a's list
    for group in prefs[a]:
        if b in group:
            group.remove(b)
            break
    prefs[a] = [g for g in prefs[a] if g]

    # Remove a from b's list
    for group in prefs[b]:
        if a in group:
            group.remove(a)
            break
    prefs[b] = [g for g in prefs[b] if g]

    # Break semi-assignments if provided
    if semi and semi_to:
        if b in semi[a]:
            semi[a].remove(b)
            semi_to[b].remove(a)
        if a in semi[b]:
            semi[b].remove(a)
            semi_to[a].remove(b)


def phase_one(prefs, agents, check_odd):
    """
    Execute Phase 1 of the algorithm, modifying prefs in place.

    Returns True if successful (or no check), False if odd number of non-empty lists when check_odd=True.
    """
    semi = defaultdict(set)
    semi_to = defaultdict(set)

    while True:
        # Proposal loop
        while True:
            free_ps = [p for p in agents if prefs[p] and not semi[p]]
            if not free_ps:
                break
            p = min(free_ps)
            f_p = get_f(prefs, p)
            for q in f_p:
                semi[p].add(q)
                semi_to[q].add(p)
                try:
                    rank_p = get_rank(prefs, q, p)
                except ValueError:
                    continue  # Skip if inconsistency
                to_reject = [r for r in list(semi_to[q]) if get_rank(prefs, q, r) > rank_p]
                for r in to_reject:
                    semi[r].discard(q)
                    semi_to[q].discard(r)
                    delete_pair(prefs, q, r, semi, semi_to)

        # Form semi-assignment graph
        active = [p for p in agents if prefs[p]]
        if not active:
            break  # Z empty

        U_suffix = '_U'
        V_suffix = '_V'
        node_to_agent = {}
        U_nodes = []
        V_nodes = []
        for p in active:
            u = str(p) + U_suffix
            v = str(p) + V_suffix
            U_nodes.append(u)
            V_nodes.append(v)
            node_to_agent[u] = p
            node_to_agent[v] = p

        nx_graph = nx.Graph()
        for node in U_nodes:
            nx_graph.add_node(node, bipartite=0)
        for node in V_nodes:
            nx_graph.add_node(node, bipartite=1)
        for p in active:
            u = str(p) + U_suffix
            for q in get_f(prefs, p):
                v = str(q) + V_suffix
                nx_graph.add_edge(u, v)

        matching = nx.bipartite.maximum_matching(nx_graph, U_nodes)
        pair = {u: matching.get(u) for u in U_nodes if u in matching}
        rev_pair = {v: u for u, v in pair.items()}
        F_U = [u for u in U_nodes if u not in pair]

        if not F_U:
            break  # Z empty

        directed = defaultdict(set)
        for u in U_nodes:
            for v in nx_graph.neighbors(u):
                if pair.get(u) != v:
                    directed[u].add(v)
        for v in V_nodes:
            u = rev_pair.get(v)
            if u:
                directed[v].add(u)

        visited = set()
        queue = deque(F_U)
        for fu in F_U:
            visited.add(fu)
        while queue:
            node = queue.popleft()
            for neigh in directed[node]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)

        Z_nodes = [node for node in visited if node.endswith(U_suffix)]
        Z = [node_to_agent[node] for node in Z_nodes]

        if not Z:
            break

        N_Z = set()
        for p in Z:
            N_Z.update(get_f(prefs, p))

        for p in N_Z:
            l_p = get_l(prefs, p)
            for q in list(l_p):
                delete_pair(prefs, p, q, semi, semi_to)

    if check_odd:
        non_empty_count = len([p for p in agents if prefs[p]])
        if non_empty_count % 2 == 1:
            return False
    return True


def find_strongly_stable_matching(prefs):
    """
    Compute a strongly stable matching for the given SRTI instance.

    Parameters:
    prefs (dict[int, list[list[int]]]): Agent to preference list (ties as sublists, most to least preferred).

    Returns:
    list[tuple[int, int]]: List of matched pairs (sorted), or None if no strongly stable matching exists.

    Raises:
    ValueError: For invalid inputs.
    """
    agents = sorted(prefs.keys())

    # Phase 1
    if not phase_one(prefs, agents, check_odd=True):
        return None

    # Phase 2
    while True:
        multi_level = [x for x in agents if len(prefs[x]) > 1]
        if not multi_level:
            break
        x = min(multi_level)

        # Branch f(x)
        prefs_f = copy.deepcopy(prefs)
        f_x = get_f(prefs_f, x)
        for z in f_x:
            l_z = get_l(prefs_f, z)
            for w in list(l_z):
                delete_pair(prefs_f, z, w)
        phase_one(prefs_f, agents, check_odd=False)
        has_empty_f = any(not prefs_f[p] for p in agents)

        # Branch l(x)
        prefs_l = copy.deepcopy(prefs)
        l_x = get_l(prefs_l, x)
        for y in list(l_x):
            delete_pair(prefs_l, x, y)
        phase_one(prefs_l, agents, check_odd=False)
        has_empty_l = any(not prefs_l[p] for p in agents)

        if not has_empty_f:
            prefs = prefs_f
        elif not has_empty_l:
            prefs = prefs_l
        else:
            return None

    # Final assignment graph
    G = nx.Graph()
    for x in agents:
        f_x = get_f(prefs, x)
        for y in f_x:
            if x > y:
                G.add_edge(x, y)

    matching = nx.max_weight_matching(G)
    pairs = []
    seen = set()
    for a in matching:
        b = matching[a]
        pair = (min(a, b), max(a, b))
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)

    return sorted(pairs)


import random
from typing import Dict, List


def generate_random_srti_prefs(
        n_agents: int,
        min_ties: int = 1,
        max_ties: int = 3,
        max_group_size: int = 3,
        seed: int  = None
) -> Dict[int, List[List[int]]]:
    """
    生成随机的SRTI偏好列表（带平局的严格偏好）。

    参数:
    n_agents (int): 代理数量，编号为 0 到 n_agents-1。
    min_ties (int): 每个偏好列表中最少平局组数。
    max_ties (int): 每个偏好列表中最多的平局组数。
    max_group_size (int): 每个平局组的最大大小。
    seed (int, optional): 随机种子，用于可复现性。

    返回:
    dict[int, list[list[int]]]: 代理到偏好列表的映射。
        每个代理的偏好是一个列表，列表元素是平局组（list），从最优到最差。

    示例:
    {
        0: [[1, 2], [3]],           # 0 最喜欢 1 和 2（平局），其次是 3
        1: [[2], [0, 3], [4]],      # 1 最喜欢 2，其次 0 和 3，最差 4
        ...
    }
    """
    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    if min_ties < 1 or max_ties < min_ties or max_group_size < 1:
        raise ValueError("Invalid tie/group parameters")

    random.seed(seed)

    prefs = {}
    agents = list(range(n_agents))

    for p in agents:
        # 随机决定平局组数量
        n_groups = random.randint(min_ties, max_ties)
        groups = []
        remaining = [a for a in agents if a != p]
        random.shuffle(remaining)

        for i in range(n_groups):
            if not remaining:
                break
            # 决定当前组大小
            group_size = min(
                random.randint(1, max_group_size),
                len(remaining)
            )
            group = remaining[:group_size]
            groups.append(sorted(group))  # 排序便于比较
            remaining = remaining[group_size:]

        # 如果还有剩余代理，放入最后一组
        if remaining:
            groups.append(sorted(remaining))

        prefs[p] = groups

    return prefs


# 示例用法
if __name__ == "__main__":
    prefs = generate_random_srti_prefs(n_agents=5, seed=42)
    for p, pref in prefs.items():
        print(f"{p}: {pref}")
