"""
FedEntOpt.py

Entropy-based client selection strategy, from:
    A. Lutz, G. Steidl, K. Muller, W. Samek,
    "Optimizing Federated Learning by Entropy-Based Client Selection", 2024.

--------------------------------------------------------------------------
WHAT THIS FILE DOES (paper -> code map)
--------------------------------------------------------------------------
The paper's Algorithm 1 works like this:

  1. Every client k has a label-count vector l^(k) in N^C: for each of the
     C classes, how many local training samples belong to that class.
  2. The server keeps a running sum L (starts at 0) of the label-count
     vectors of the clients it has picked *so far, in the current round*.
  3. The server picks a first client k0 uniformly at random, sets L = l^(k0).
  4. It then repeatedly adds ONE more client: whichever remaining client m
     maximizes the Shannon entropy H( (L + l^(m)) / ||L + l^(m)||_1 ),
     i.e. whichever client's data best "fills in" the classes the current
     subset is missing.
  5. A FIFO buffer B of size Q temporarily blocks recently-picked clients,
     so the server doesn't pick the exact same subset every round.
     Buffer size must satisfy 0 < Q <= K - M (K = pool size, M = clients
     picked per round), or there would be nothing left to pick from.

--------------------------------------------------------------------------
WHY THE CODE LOOKS A BIT MORE COMPLEX THAN THE ALGORITHM
--------------------------------------------------------------------------
Look at FederatedStrategy.py in the bundle:

    host_select_train_clients(self, ready_clients: List[ClientId])

This only receives a list of client IDs. It has NO way to see label data,
because client selection happens on the server, and clients haven't
uploaded anything yet at that point. So the server does not know any
l^(k) the first time it sees a client.

Fix (same trick SecureAggregation.py in the bundle uses for its own
one-time key-exchange step): a client attaches its label-count vector
to its upload exactly ONCE, the first time it is ever selected and
trained. The server caches it in self._client_label_counts, keyed by
client_id, and reuses it in every later round.

Consequence: for any client the server has never heard from yet, there
is no l^(k) to use in the entropy formula. Such "unknown" clients are
selected uniformly at random (this matches the paper's own treatment of
the very first client k0, which is also chosen uniformly at random,
since L = 0 before it is picked -- entropy of an all-zero vector is
undefined). Once every client in the pool has been seen once, selection
follows Algorithm 1 exactly, every round.
"""

import numpy as np
from collections import deque

from ..config.configuration import ConfigurationManager
from .FedAvg import FedAvg


class FedEntOpt(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---------------- client-side state ----------------
        # l^(k): this client's own label-count vector, computed once from
        # its local training labels. None on the server, since only
        # clients hold self.train_data.
        if getattr(self, 'train_data', None) is not None:
            self._label_counts = self._compute_label_counts(self.train_data['y'])
        else:
            self._label_counts = None
        self._has_reported_labels = False  # have we sent l^(k) to the server yet?

        # ---------------- server-side state -----------------
        self._client_label_counts = {}    # client_id -> np.ndarray, shape (C,); this is how the server learns l^(k) for each client
        self._selection_buffer = deque()  # FIFO buffer B from the paper
        self._buffer_q = None             # Q, computed once K (pool size) is known

    # ====================================================================
    # Small helper functions
    # ====================================================================
    @staticmethod
    def _compute_label_counts(y):
        """
        Build l^(k) in N^C from this client's label array y.
        Handles both one-hot labels (shape [N, C]) and integer class
        labels (shape [N]).
        """
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[-1] > 1:
            # one-hot / soft labels: sum contributions of each class column
            counts = y.reshape(-1, y.shape[-1]).sum(axis=0)
        else:
            y_flat = y.reshape(-1).astype(int)
            mdl_cfg = ConfigurationManager().model_config
            num_classes = getattr(mdl_cfg, 'num_classes', None) or (int(y_flat.max()) + 1)
            counts = np.bincount(y_flat, minlength=num_classes).astype(np.float64)
        return counts

    def _get_buffer_size(self, k_total, m_select):
        """
        Q must satisfy 0 < Q <= K - M (paper, Sec. IV).
        Default: 70% of the pool size, then clipped into the valid range.
        Override by setting strategy_config['fedentopt_buffer_fraction']
        in your yaml config (0.0, 1.0].
        """
        strategy_cfg = getattr(ConfigurationManager().model_config, 'strategy_config', {}) or {}
        q_fraction = strategy_cfg.get('fedentopt_buffer_fraction', 0.7)
        q = int(round(q_fraction * k_total))
        return max(1, min(q, max(1, k_total - m_select)))

    @staticmethod
    def _entropy(vec):
        """
        Shannon entropy H(p), base 2, of vec after normalizing it into a
        probability distribution p = vec / sum(vec). Matches eq. (5)/(6)
        in the paper: H(p) = - sum_c p_c * log2(p_c).
        """
        total = vec.sum()
        if total <= 0:
            return -np.inf
        p = vec / total
        p = p[p > 0]  # 0 * log2(0) is defined as 0, so just drop zero terms
        return float(-(p * np.log2(p)).sum())

    # ====================================================================
    # Client side: attach l^(k) to the first upload only
    # ====================================================================
    def retrieve_local_upload_info(self):
        params = super().retrieve_local_upload_info()
        if not self._has_reported_labels and self._label_counts is not None:
            self._has_reported_labels = True
            return {'params': params, 'label_counts': self._label_counts, 'client_id': self.client_id}
        return {'params': params, 'label_counts': None, 'client_id': self.client_id}

    # ====================================================================
    # Host side: cache label counts, then aggregate params as usual
    # ====================================================================
    def update_host_params(self, client_params, aggregate_weights):
        # client_params is a list of the dicts returned above, one per
        # client that trained this round.
        for entry in client_params:
            if entry.get('label_counts') is not None:
                self._client_label_counts[entry['client_id']] = np.asarray(entry['label_counts'])
        model_params_only = [entry['params'] for entry in client_params]
        super().update_host_params(model_params_only, aggregate_weights)

    # ====================================================================
    # Host side: Algorithm 1 -- entropy-maximizing client selection
    # ====================================================================
    def host_select_train_clients(self, ready_clients):
        cfg = ConfigurationManager()
        m_select = cfg.num_of_train_clients_contacted_per_round
        ready_clients = list(ready_clients)
        k_total = len(ready_clients)

        if self._buffer_q is None:
            self._buffer_q = self._get_buffer_size(k_total, m_select)

        known = [c for c in ready_clients if c in self._client_label_counts]
        unknown = [c for c in ready_clients if c not in self._client_label_counts]

        selected = []

        # --- Bootstrap step: clients we've never heard from must be
        # trained at least once so we learn their l^(k) for future rounds.
        # Picked uniformly at random, same spirit as the paper's k0. ---
        if unknown:
            n_bootstrap = min(len(unknown), m_select)
            selected.extend(np.random.choice(unknown, n_bootstrap, replace=False).tolist())

        remaining_slots = m_select - len(selected)

        if remaining_slots > 0 and known:
            buffered = set(self._selection_buffer)
            candidates = [c for c in known if c not in buffered and c not in selected]
            if not candidates:
                # Buffer currently blocks the entire known pool -> ignore
                # it this round rather than fail to fill the quota.
                candidates = [c for c in known if c not in selected]

            C = next(iter(self._client_label_counts.values())).shape[0]
            L = np.zeros(C)
            for c in selected:
                if c in self._client_label_counts:
                    L += self._client_label_counts[c]

            # Algorithm 1, line 6-7: if this round's subset is still empty,
            # the first pick is uniform at random (L is all zeros, entropy
            # of L + l^(m) alone would just favor whichever l^(m) happens
            # to have the most samples -- not what the paper intends).
            if not selected and candidates:
                first = candidates[np.random.randint(len(candidates))]
                selected.append(first)
                L += self._client_label_counts[first]
                candidates.remove(first)
                remaining_slots -= 1

            # Algorithm 1, line 8-9: greedily add the client that maximizes
            # entropy of the combined, normalized label distribution.
            while remaining_slots > 0 and candidates:
                best_client, best_entropy = None, -np.inf
                for c in candidates:
                    h = self._entropy(L + self._client_label_counts[c])
                    if h > best_entropy:
                        best_entropy, best_client = h, c
                selected.append(best_client)
                L += self._client_label_counts[best_client]
                candidates.remove(best_client)
                remaining_slots -= 1

        # Edge case: small pools / lots of clients still unknown -> pad
        # with random ready clients so we always return exactly m_select.
        if len(selected) < m_select:
            leftover = [c for c in ready_clients if c not in selected]
            n_pad = min(m_select - len(selected), len(leftover))
            if n_pad > 0:
                selected.extend(np.random.choice(leftover, n_pad, replace=False).tolist())

        # Algorithm 1, line 10-12: push newly picked clients into the FIFO
        # buffer B, then evict the oldest entries once |B| > Q.
        for c in selected:
            self._selection_buffer.append(c)
        while len(self._selection_buffer) > self._buffer_q:
            self._selection_buffer.popleft()

        self.train_selected_clients = selected
        return self.train_selected_clients