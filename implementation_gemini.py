import copy
import networkx as nx
from typing import List, Dict, Set, Any, Tuple, Optional
import random
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


# 备注：此算法实现依赖 'networkx' 库。
# 请确保已安装: pip install networkx

class SRTIStrongMatcher:
    """
    根据 "Phase 1" 和 "Phase 2" 伪代码实现 SRTI-strong 算法。

    该算法用于寻找一个强稳定匹配（Strongly Stable Matching），
    特别适用于偏好列表中可能存在平局（ties）的情况。

    算法的核心数据结构是一个偏好表 "T"，在 Python 中实现为
    一个字典，其中键是参与者 (agent)，值是偏好列表。
    每个偏好列表是一个列表的列表，例如：
    prefs = {
        'a': [['b', 'c'], ['d']],  # 'a' 最偏好 'b' 和 'c' (平局)，其次是 'd'
        'b': [['a'], ['c', 'd']],
        'c': [['a', 'b'], ['d']],
        'd': [['a'], ['b'], ['c']]
    }
    """

    def __init__(self, initial_prefs: Dict[Any, List[List[Any]]]):
        """
        初始化算法。

        参数:
        initial_prefs (dict): 初始偏好表。
            格式: {agent: [[agent1, agent2], [agent3], ...]}
        """
        if not isinstance(initial_prefs, dict):
            raise ValueError("初始偏好 'initial_prefs' 必须是一个字典。")

        self.initial_prefs = copy.deepcopy(initial_prefs)
        self.agents = set(initial_prefs.keys())
        # 确保所有在偏好列表中的 agent 也在主键中
        all_agents_in_lists = set()
        for pref_list in initial_prefs.values():
            for tied_set in pref_list:
                all_agents_in_lists.update(tied_set)

        missing_agents = all_agents_in_lists - self.agents
        if missing_agents:
            # 如果某些 agent 只出现在列表中但没有自己的列表，为它们添加空列表
            for agent in missing_agents:
                if agent not in self.initial_prefs:
                    self.initial_prefs[agent] = []
            self.agents.update(missing_agents)

        # 建立一个总序，用于最终图的构建
        self.agent_order = sorted(list(self.agents))
        self.agent_order_map = {agent: i for i, agent in enumerate(self.agent_order)}

    def _build_ranks(self, prefs: Dict) -> Dict[Any, Dict[Any, int]]:
        """
        从偏好表构建一个等级（rank）查找字典。
        等级越低（例如 0）表示偏好越高。
        """
        ranks = {}
        for agent, pref_list in prefs.items():
            ranks[agent] = {}
            for rank, tied_agents in enumerate(pref_list):
                for preferred_agent in tied_agents:
                    ranks[agent][preferred_agent] = rank
        return ranks

    def _prefers(self, agent: Any, p1: Any, p2: Any, ranks: Dict) -> bool:
        """
        检查 'agent' 是否偏好 'p1' 超过 'p2'。
        """
        agent_ranks = ranks.get(agent, {})
        rank1 = agent_ranks.get(p1, float('inf'))
        rank2 = agent_ranks.get(p2, float('inf'))
        return rank1 < rank2

    def _get_head(self, agent: Any, prefs: Dict) -> List[Any]:
        """获取 agent 偏好列表的头部 f(agent)"""
        if agent in prefs and prefs[agent]:
            return prefs[agent][0]
        return []

    def _get_tail(self, agent: Any, prefs: Dict) -> List[Any]:
        """获取 agent 偏好列表的尾部 l(agent)"""
        if agent in prefs and prefs[agent]:
            return prefs[agent][-1]
        return []

    def _get_non_head(self, agent: Any, prefs: Dict) -> List[Any]:
        """获取 agent 偏好列表中所有不在头部的 agent (即 'tail of p's list')"""
        flat_list = []
        if agent in prefs and len(prefs[agent]) > 1:
            for pref_set in prefs[agent][1:]:
                flat_list.extend(pref_set)
        return flat_list

    def _is_list_empty(self, agent: Any, prefs: Dict) -> bool:
        """检查 agent 的偏好列表是否为空"""
        return not prefs.get(agent, [])

    def _get_active_agents(self, prefs: Dict) -> Set[Any]:
        """获取所有偏好列表非空的 agent"""
        return {agent for agent in self.agents if not self._is_list_empty(agent, prefs)}

    def _delete_pair(self, p: Any, q: Any, prefs: Dict,
                     semi_assigned: Dict, ranks: Dict) -> bool:
        """
        从偏好列表中删除 {p, q} 对，并更新 ranks 和 semi_assigned。
        修改是 *in-place* 的。
        """
        deleted = False

        # 从 p 的列表中移除 q
        if p in prefs and q in ranks.get(p, {}):
            new_p_list = []
            for pref_set in prefs[p]:
                # 过滤掉 q
                new_set = [agent for agent in pref_set if agent != q]
                if new_set:  # 仅保留非空集合
                    new_p_list.append(new_set)

            if len(new_p_list) < len(prefs[p]) or any(len(old) != len(new) for old, new in zip(prefs[p], new_p_list)):
                deleted = True
            prefs[p] = new_p_list

        # 从 q 的列表中移除 p
        if q in prefs and p in ranks.get(q, {}):
            new_q_list = []
            for pref_set in prefs[q]:
                # 过滤掉 p
                new_set = [agent for agent in pref_set if agent != p]
                if new_set:  # 仅保留非空集合
                    new_q_list.append(new_set)

            if len(new_q_list) < len(prefs[q]) or any(len(old) != len(new) for old, new in zip(prefs[q], new_q_list)):
                deleted = True
            prefs[q] = new_q_list

        # 打破半分配
        if q in semi_assigned.get(p, set()):
            semi_assigned[p].remove(q)
            deleted = True
        if p in semi_assigned.get(q, set()):
            semi_assigned[q].remove(p)
            deleted = True

        if deleted:
            # 如果发生了删除，重建 ranks
            # (注意：这是一个原地更新)
            ranks.clear()
            ranks.update(self._build_ranks(prefs))

        return deleted

    def _build_semi_assignment_graph(self, prefs: Dict, active_agents: Set, visualize: bool=False) -> nx.Graph:
        """
        构建半分配图 G=(U, V, E)。
        U (bipartite=0) 是提议节点，V (bipartite=1) 是接收节点。
        """
        G = nx.Graph()
        # 节点名称使用 'u_' 和 'v_' 前缀以区分 U 和 V
        U_nodes = {f"u_{p}" for p in active_agents}
        V_nodes = {f"v_{p}" for p in active_agents}
        G.add_nodes_from(U_nodes, bipartite=0)
        G.add_nodes_from(V_nodes, bipartite=1)

        for p in active_agents:
            head_p = self._get_head(p, prefs)
            for q in head_p:
                if q in active_agents:
                    # 添加从 p (U) 到 q (V) 的边
                    G.add_edge(f"u_{p}", f"v_{q}")
        if visualize:
            # Define positions using bipartite layout
            pos = nx.bipartite_layout(G, U_nodes)


            U_nodes = {f"u_{p}" for p in active_agents}
            node_colors = ["#1f78b4" if node in U_nodes else "#33a02c" for node in G.nodes()]  # Blue for U, Green for V

            # 4. Draw the graph components separately
            #    Draw nodes with increased size
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
            #    Draw edges
            nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray", alpha=0.6)
            #    Draw labels with enhanced readability
            nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")

            # 5. Final touches
            plt.axis("off")  # Turn off the axes for clarity
            plt.tight_layout()  # Adjust layout to minimize padding
            plt.title("Semi-Assignment Graph", fontsize=8)
            plt.show()

        return G

    def _find_critical_set(self, G: nx.Graph, active_agents: Set) -> Tuple[Set, Set]:
        """
        在半分配图 G 中找到唯一的最小临界集 Z (U 的子集)。
        Z 是具有最大缺陷 delta(Z) = |Z| - |N(Z)| 的集合。
        """
        U_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
        V_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 1}

        if not U_nodes:
            return set(), set()  # 图为空
        # TODO: double check correctness of deficiency here
        # 1. 找到一个最大匹配 M
        matching = nx.bipartite.maximum_matching(G, top_nodes=U_nodes)

        # 2. 找到 U 中未匹配的节点 U_unmatched
        U_unmatched = U_nodes - set(matching.keys())

        # 3. 找到从 U_unmatched 出发可通过 M-交错路径到达的所有节点
        # 我们构建一个有向图 D：
        # - 如果 (u, v) in M，添加 v -> u
        # - 如果 (u, v) in E \ M，添加 u -> v
        D = nx.DiGraph()
        for u in U_nodes:
            for v in G.neighbors(u):
                if matching.get(u) == v:
                    D.add_edge(v, u)  # M 中的边，方向反转
                else:
                    D.add_edge(u, v)  # 不在 M 中的边，方向 u -> v

        # 4. 从 U_unmatched 开始进行图遍历
        reachable_nodes = set(U_unmatched)
        for start_node in U_unmatched:
            reachable_nodes.update(nx.descendants(D, start_node))

        Z_nodes = reachable_nodes.intersection(U_nodes)
        N_Z_nodes = reachable_nodes.intersection(V_nodes)  # N(Z) = A \cap V

        # 5. Z 是 A \cap U, N(Z) 是 A \cap V
        # 转换回 agent 名称
        Z = {n.split('_', 1)[1] for n in Z_nodes}
        N_Z = {n.split('_', 1)[1] for n in N_Z_nodes}

        return Z, N_Z

    def _stabilize(self, prefs: Dict, semi_assigned: Dict) -> Tuple[Dict, Dict, Set]:
        """
        执行 Phase 1 的主循环 (Alg. 1, lines 2-17)。
        这是一个稳定过程，会修改 prefs 和 semi_assigned。

        返回:
        (stabilized_prefs, stabilized_ranks, active_agents)
        """
        # 创建副本以避免修改原始输入
        local_prefs = copy.deepcopy(prefs)
        local_semi = copy.deepcopy(semi_assigned)
        local_ranks = self._build_ranks(local_prefs)
        active_agents = self._get_active_agents(local_prefs)

        # 进度检测
        prev_state = None
        iteration = 0
        max_iterations = len(active_agents)**2  # 防御性上限，可根据实例规模调整

        while True:  # Repeat...Until Z = \emptyset
            iteration += 1
            if iteration > max_iterations:
                logger.error("Exceeded max iterations in _stabilize (%d). Breaking to avoid infinite loop.",
                             max_iterations)
                break
            # (Lines 3-10) While some agent p is free...
            # 我们使用一个栈来进行迭代，因为打破半分配可能会使 agent 重新变为 free
            free_agents_stack = list({p for p in active_agents if not local_semi.get(p)})

            while free_agents_stack:
                p = free_agents_stack.pop()

                # 检查 p 是否仍然 free 且 active
                if p not in active_agents or local_semi.get(p):
                    continue

                head_p = self._get_head(p, local_prefs)
                if not head_p:
                    continue  # 列表为空，保持 free

                for q in head_p:
                    if q not in active_agents:
                        continue

                    local_semi.setdefault(p, set()).add(q)  # (Line 5)

                    # (Line 6) 检查 q 的列表
                    # 创建一个快照，因为 local_ranks[q] 可能会在循环中被修改
                    agents_on_q_list = list(local_ranks.get(q, {}).keys())

                    for r in agents_on_q_list:
                        if r not in active_agents:
                            continue

                        if self._prefers(q, p, r, local_ranks):
                            # (Line 7) If r is semi-assigned to q
                            if r in local_semi.get(q, set()):
                                local_semi[q].remove(r)  # (Line 8)
                                # q 变为 free
                                if not local_semi[q]:
                                    free_agents_stack.append(q)

                            # (Line 9) delete the pair {q, r}
                            # _delete_pair 会原地更新 local_prefs, local_semi, local_ranks
                            if self._delete_pair(q, r, local_prefs, local_semi, local_ranks):
                                # 列表发生变化，更新 active_agents
                                active_agents = self._get_active_agents(local_prefs)

            # (Line 11) Form the semi-assignment graph
            G_semi = self._build_semi_assignment_graph(local_prefs, active_agents)
            # (Line 12) Find the critical set Z
            if not G_semi.nodes():
                Z, N_Z = set(), set()
            else:
                Z, N_Z = self._find_critical_set(G_semi, active_agents)
            logger.info("Critical set Z and N_z len are %d and %d respectively, Z=%s, N_Z=%s", len(Z), len(N_Z), Z,
                         N_Z)
            snapshot_lines = [f"Prefs snapshot (iter {iteration}):"]
            for agent in sorted(local_prefs):
                snapshot_lines.append(f"{agent}: {local_prefs[agent]}")
            logger.info("\n".join(snapshot_lines))

            # (Line 17) Until Z = \emptyset
            if not Z:
                break

            # (Lines 13-16) Delete pairs from N(Z)'s tail
            deleted_something = False
            # 收集所有要删除的对，以避免在迭代时修改
            pairs_to_delete = []
            for p in N_Z:
                non_head_agents = self._get_non_head(p, local_prefs)
                for q in non_head_agents:
                    pairs_to_delete.append((p, q))

            for p, q in pairs_to_delete:
                if self._delete_pair(p, q, local_prefs, local_semi, local_ranks):
                    deleted_something = True

            if deleted_something:
                active_agents = self._get_active_agents(local_prefs)

            # 论文保证如果 Z 非空，则 N_Z 中至少有一个 agent
            # 的 non-head 列表非空，因此删除总会发生，避免了死循环。

        return local_prefs, local_ranks, active_agents

    def _run_phase2(self, T1_prefs: Dict, T1_ranks: Dict, T1_active_agents: Set) -> Tuple[
        Optional[Dict], Optional[str]]:
        """
        执行 Phase 2 (Alg. 2)。

        返回: (final_prefs, error_message)
        """
        T = copy.deepcopy(T1_prefs)
        active_agents = T1_active_agents.copy()

        # (Line 2) While some agent x has f_T(x) != l_T(x)
        while True:
            x = None
            # 寻找一个 x，其 f(x) != l(x) (即列表包含多个偏好等级)
            # 使用排序后的列表保证确定性
            for agent in sorted(list(active_agents)):
                if not self._is_list_empty(agent, T) and len(T[agent]) > 1:
                    x = agent
                    break

            if x is None:
                # 循环终止：所有 agent 的 f(x) == l(x) 或列表为空
                break

            # (Lines 3-5) 准备 T^{f(x)}
            T_f_x = copy.deepcopy(T)
            semi_f_x = {p: set() for p in self.agents}
            ranks_f_x = self._build_ranks(T_f_x)

            pairs_to_delete_f = []
            head_x = self._get_head(x, T_f_x)
            for z in head_x:
                if z in active_agents:
                    tail_z = self._get_tail(z, T_f_x)
                    for w in tail_z:
                        pairs_to_delete_f.append((z, w))

            for z, w in pairs_to_delete_f:
                self._delete_pair(z, w, T_f_x, semi_f_x, ranks_f_x)

            # (Line 5) apply main loop of phase 1
            T_f_x_stable, _, active_f_x = self._stabilize(T_f_x, semi_f_x)

            # (Lines 6-8) 准备 T^{l(x)}
            T_l_x = copy.deepcopy(T)
            semi_l_x = {p: set() for p in self.agents}
            ranks_l_x = self._build_ranks(T_l_x)

            pairs_to_delete_l = []
            tail_x = self._get_tail(x, T_l_x)
            for y in tail_x:
                pairs_to_delete_l.append((x, y))

            for x_del, y_del in pairs_to_delete_l:
                self._delete_pair(x_del, y_del, T_l_x, semi_l_x, ranks_l_x)

            # (Line 8) apply main loop of phase 1
            T_l_x_stable, _, active_l_x = self._stabilize(T_l_x, semi_l_x)

            # (Lines 9-15) Check conditions
            f_x_has_empty = any(self._is_list_empty(p, T_f_x_stable) for p in active_f_x)
            l_x_has_empty = any(self._is_list_empty(p, T_l_x_stable) for p in active_l_x)

            if not f_x_has_empty:
                T = T_f_x_stable
                active_agents = active_f_x
            elif not l_x_has_empty:
                T = T_l_x_stable
                active_agents = active_l_x
            else:
                # (Line 14) report no strongly stable matching
                return None, "No strongly stable matching exists (Phase 2 failure)"

        # (Line 17) 返回最终稳定的 T
        return T, None

    def _build_final_graph_and_match(self, T_final: Dict, active_agents: Set) -> Tuple[
        Optional[Set[Tuple]], Optional[str]]:
        """
        构建最终分配图 (Alg. 2, Line 17)
        并找到一个完美匹配 (Alg. 2, Line 18)。
        """
        G_final = nx.Graph()
        G_final.add_nodes_from(active_agents)

        # (Item 2) for each agent x, add edge (x,y) if y in f(x) and x > y
        for x in active_agents:
            head_x = self._get_head(x, T_final)
            for y in head_x:
                if y in active_agents and self.agent_order_map[x] > self.agent_order_map[y]:
                    G_final.add_edge(x, y)

        try:
            # 找到一个最大基数匹配（即最大权重匹配，权重全为1）
            matching = nx.max_weight_matching(G_final, maxcardinality=True)

            # 检查是否为完美匹配
            if len(matching) * 2 != len(active_agents):
                # 根据引理 3.3.4，这不应该发生
                return None, f"Final graph does not contain a perfect matching. (Nodes: {len(active_agents)}, Matched: {len(matching) * 2})"

            # (Line 19) output M
            return matching, None

        except Exception as e:
            return None, f"Error finding perfect matching in final graph: {e}"




    def find_matching(self) -> Tuple[Optional[Set[Tuple]], str]:
        """
        运行 SRTI-strong 算法 (Phase 1 和 Phase 2) 寻找强稳定匹配。

        返回:
        (matching, message)
        'matching' 是一个元组的集合，代表匹配的对。
        如果找不到匹配，'matching' 为 None，'message' 包含原因。
        """

        # --- Phase 1 (Alg. 1) ---

        # (Line 1) initialize
        prefs_T1 = copy.deepcopy(self.initial_prefs)
        semi_assigned_T1 = {p: set() for p in self.agents}

        # (Lines 2-17) Main loop
        T1_stable, T1_ranks, active_agents_T1 = self._stabilize(prefs_T1, semi_assigned_T1)

        # (Lines 18-20) Check for odd number of agents
        if len(active_agents_T1) % 2 != 0:
            return None, "No strongly stable matching exists (Phase 1: odd number of agents with non-empty lists)"

        # --- Phase 2 (Alg. 2) ---

        # (Line 1) T <- T^1
        T_final, error = self._run_phase2(T1_stable, T1_ranks, active_agents_T1)

        if error:
            return None, error

        # (Lines 17-19) Final graph and matching
        active_final = self._get_active_agents(T_final)

        # 再次检查奇偶性，以防 Phase 2 产生了奇数个 active agent
        if len(active_final) % 2 != 0:
            return None, "No strongly stable matching exists (Phase 2: final set of agents is odd)"

        if not active_final:
            return set(), "Strongly stable matching found (empty matching)"  # 空匹配是稳定的

        matching, error = self._build_final_graph_and_match(T_final, active_final)

        if error:
            return None, error

        return matching, "Strongly stable matching found"


def generate_random_prefs(num_agents: int, max_tie_size: Optional[int] = None) -> Dict[str, List[List[str]]]:
    """
    生成一个随机的偏好表（prefs）字典，用于测试 SRTIStrongMatcher。

    参数:
    num_agents (int): 参与者的数量。
    max_tie_size (int, optional):
        单个平局集合中的最大参与者数量。
        如果为 1，则所有偏好都是严格的（没有平局）。
        如果为 None，平局大小可以是从 1 到 (num_agents - 1) 的任何值。

    返回:
    Dict[str, List[List[str]]]:
        随机生成的偏好表，格式为:
        {agent: [[agent1, agent2], [agent3], ...]}

    异常:
    ValueError: 如果 num_agents < 1。
    """
    if num_agents < 1:
        raise ValueError("参与者数量 (num_agents) 必须至少为 1。")

    if max_tie_size is not None and max_tie_size < 1:
        logger.warning("警告: max_tie_size (%d) 小于 1。将其设置为 1（严格偏好）。", max_tie_size)
        max_tie_size = 1

    agents = [f'agent_{i + 1}' for i in range(num_agents)]
    prefs = {}

    for agent in agents:
        # 创建一个包含所有其他参与者的列表
        other_agents = [a for a in agents if a != agent]
        random.shuffle(other_agents)

        pref_list = []
        current_index = 0

        while current_index < len(other_agents):
            remaining_agents = len(other_agents) - current_index

            # 确定此平局集合的上限
            if max_tie_size is None:
                # 如果未指定，上限是所有剩余的 agent
                upper_bound = remaining_agents
            else:
                # 否则，上限是 max_tie_size 和剩余 agent 数量中的较小者
                upper_bound = min(max_tie_size, remaining_agents)

            # 随机选择一个大小
            # 确保大小至少为 1
            tie_size = random.randint(1, upper_bound)

            # 从打乱的列表中切片
            tied_set = other_agents[current_index: current_index + tie_size]
            pref_list.append(tied_set)

            # 推进索引
            current_index += tie_size

        prefs[agent] = pref_list

    return prefs
if __name__ == '__main__':
    # 配置根日志记录器，确保 INFO 级别及以上会输出到控制台
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    # 将模块级 logger 的级别显式设置为 INFO（保障在特殊配置下也能输出）
    logger.setLevel(logging.INFO)
    prefs=generate_random_prefs(num_agents=8, max_tie_size=3)
    # python
    snapshot_lines = ["Generated prefs:"]
    for agent in sorted(prefs):
        snapshot_lines.append(f"{agent}: {prefs[agent]}")
    logger.info("\n".join(snapshot_lines))
    SRTI=SRTIStrongMatcher(prefs)
    matching,_= SRTI.find_matching()
    logger.info("Matching: %s", matching)