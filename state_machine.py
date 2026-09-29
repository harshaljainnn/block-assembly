"""
Assembly State Machine for Block Assembly Quality Inspection.
Enforces sequential assembly progression, implements rolling-window temporal consensus smoothing,
and strictly latches diagnostic errors when out-of-order steps or spatial violations occur.
Adapted from the battle-tested Pen Assembly Quality Inspection System.
"""

from collections import deque, Counter
from block_config import ASSEMBLY_STATES, STEP_TITLES


class AssemblyStateMachine:
    def __init__(self, state_order=None, window_size=10, min_consensus=6):
        """
        state_order: ordered list of state names. Defaults to ASSEMBLY_STATES.
        window_size: number of recent frames to buffer for majority voting (~0.33s @ 30fps).
        min_consensus: minimum frame votes required to confirm state transition.
        """
        self.state_order = state_order or ASSEMBLY_STATES
        self.current_index = 0
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.history = deque(maxlen=window_size)

        # Strict error tracking
        self.error_active = False
        self.error_detail = ""
        self.skipped_step_index = None

    def reset(self):
        """Resets assembly tracker back to Step 0."""
        self.current_index = 0
        self.history.clear()
        self.error_active = False
        self.error_detail = ""
        self.skipped_step_index = None

    def get_consensus(self):
        """Returns the most frequent valid state in the rolling window and its vote count."""
        valid_votes = [s for s in self.history if s is not None and s in self.state_order]
        if not valid_votes:
            return None, 0
        counts = Counter(valid_votes)
        top_state, count = counts.most_common(1)[0]
        return top_state, count

    def get_step_title(self, index):
        if 0 <= index < len(self.state_order):
            s = self.state_order[index]
            return STEP_TITLES.get(s, s)
        return "Complete"

    def update_smoothed(self, raw_state, is_valid_spatial=True, diagnostic=""):
        """
        Pushes a new frame prediction into the rolling buffer, performs majority voting,
        and triggers state advance or error latching once consensus is reached.
        """
        self.history.append(raw_state)
        consensus_state, votes = self.get_consensus()
        ratio = f"{votes}/{len(self.history)}"

        # If spatial constraints are violated, latch error immediately
        if not is_valid_spatial and diagnostic and not diagnostic.startswith("PASS"):
            self.error_active = True
            self.error_detail = diagnostic
            return "error", self.error_detail, consensus_state, ratio

        if consensus_state is None or votes < self.min_consensus:
            if self.error_active:
                return "error", self.error_detail, consensus_state, ratio
            return "holding", "stabilizing", consensus_state, ratio

        status, detail = self.update(consensus_state)
        return status, detail, consensus_state, ratio

    def current_state(self):
        return self.state_order[self.current_index]

    def update(self, matched_state):
        """
        Processes a consensus state prediction and enforces strict sequential assembly.
        """
        if matched_state is None:
            if self.error_active:
                return "error", self.error_detail
            return "holding", None

        if matched_state not in self.state_order:
            self.error_active = True
            self.error_detail = f"Unrecognized state: {matched_state}"
            return "error", self.error_detail

        matched_index = self.state_order.index(matched_state)

        # Case 1: Same as current step
        if matched_index == self.current_index:
            if self.error_active:
                return "error", self.error_detail
            return "holding", None

        # Case 2: Valid sequential advance to the EXACT next step
        elif matched_index == self.current_index + 1:
            self.current_index = matched_index
            self.error_active = False
            self.error_detail = ""
            self.skipped_step_index = None
            return "advanced", self.current_state()

        # Case 3: Reverted to an earlier step
        elif matched_index < self.current_index:
            self.error_active = True
            self.error_detail = (
                f"REVERTED to [{self.get_step_title(matched_index)}] "
                f"(was on [{self.get_step_title(self.current_index)}])"
            )
            return "error", self.error_detail

        # Case 4: Skipped ahead out of order!
        else:
            self.error_active = True
            self.skipped_step_index = self.current_index + 1
            expected = self.get_step_title(self.current_index + 1)
            detected = self.get_step_title(matched_index)
            self.error_detail = f"SKIPPED STEP! Missing [{expected}], but found [{detected}]"
            return "error", self.error_detail

    def get_steps_for_hud(self):
        """
        Returns the 9-step sequential inspection checklist for HUD rendering.
        """
        steps = []
        for i, sname in enumerate(self.state_order):
            title = STEP_TITLES.get(sname, sname)
            if i < self.current_index:
                st = "verified"
            elif self.error_active and i == self.skipped_step_index:
                st = "skipped"
            elif i == self.current_index:
                st = "error_current" if self.error_active else "current"
            else:
                st = "pending"
            steps.append({"title": title, "state": sname, "status": st, "index": i})
        return steps

    def is_complete(self):
        return self.current_index == len(self.state_order) - 1 and not self.error_active
