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
        self.skip_streak = 0
        self.empty_workspace_frames = 0
        self.completion_frames = 0
        self.empty_after_complete_frames = 0
        self.reset_flash = 0

    def reset(self):
        """Resets assembly tracker back to Step 0."""
        self.current_index = 0
        self.history.clear()
        self.error_active = False
        self.error_detail = ""
        self.skipped_step_index = None
        self.skip_streak = 0
        self.empty_workspace_frames = 0
        self.completion_frames = 0
        self.empty_after_complete_frames = 0
        self.reset_flash = 30

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
        Includes cycle auto-reset upon completion.
        """
        self.history.append(raw_state)
        consensus_state, votes = self.get_consensus()
        ratio = f"{votes}/{len(self.history)}"

        # Workspace clear tracking (auto-reset when table is cleared)
        if raw_state == "state_0_unstarted":
            self.empty_workspace_frames += 1
        else:
            self.empty_workspace_frames = 0

        # Auto-reset check when assembly cycle is completed
        if self.is_complete():
            self.completion_frames += 1
            if self.empty_workspace_frames >= 20 or self.completion_frames >= 120:
                self.reset()
                return "reset", "Cycle Complete - Ready for Next Unit", "state_0_unstarted", "1/1"

        # Mid-assembly or Error Reset:
        # If user clears desk for ~0.8s (25 frames) or during error for ~0.5s (15 frames), AUTO-RESET to Step 0
        if self.empty_workspace_frames >= (15 if self.error_active else 25):
            if self.current_index > 0 or self.error_active:
                self.reset()
                return "reset", "Workspace Cleared - Reset to Step 0", "state_0_unstarted", "1/1"

        # If assembling (parts are present on table but not yet attached), provide helpful prompt without sequence error
        if diagnostic and diagnostic.startswith("ASSEMBLING:"):
            self.error_active = False
            self.error_detail = ""
            return "assembling", diagnostic, consensus_state, ratio

        # If spatial constraints are violated or wrong parts introduced, latch error
        if not is_valid_spatial and diagnostic and not diagnostic.startswith("PASS"):
            self.error_active = True
            self.error_detail = diagnostic
            return "error", self.error_detail, consensus_state, ratio

        # If assembly is in progress and workspace is temporarily clear (e.g. assembly in hand), hold without resetting
        if consensus_state == "state_0_unstarted" and self.current_index > 0:
            return "holding", "Assembly in hand / Workspace clear", self.current_state(), ratio

        if consensus_state is None or votes < self.min_consensus:
            if self.error_active:
                return "error", self.error_detail, consensus_state, ratio
            return "holding", "stabilizing", consensus_state, ratio

        status, detail = self.update(consensus_state)
        return status, detail, consensus_state, ratio

    def current_state(self):
        return self.state_order[self.current_index]

    def update(self, matched_state, is_valid_spatial=True, diagnostic=""):
        """
        Processes a state prediction and enforces strict sequential assembly.
        Includes self-healing when frame returns to the valid current step.
        """
        if diagnostic and diagnostic.startswith("ASSEMBLING:"):
            self.error_active = False
            self.error_detail = ""
            return "assembling", diagnostic

        if not is_valid_spatial and diagnostic and not diagnostic.startswith("PASS"):
            self.error_active = True
            self.error_detail = diagnostic
            return "error", self.error_detail

        if matched_state is None:
            if self.error_active:
                return "error", self.error_detail
            return "holding", None

        if matched_state not in self.state_order:
            self.error_active = True
            self.error_detail = f"Unrecognized state: {matched_state}"
            return "error", self.error_detail

        matched_index = self.state_order.index(matched_state)

        # Case 0: Empty workspace while assembly is already in progress (assembly in hand)
        if matched_index == 0 and self.current_index > 0:
            return "holding", "Assembly in hand / Workspace clear"

        # Case 1: Same as current step - Self-healing clears transient errors
        if matched_index == self.current_index:
            self.error_active = False
            self.error_detail = ""
            self.skipped_step_index = None
            self.skip_streak = 0
            return "holding", None

        # Case 2: Valid sequential advance to the EXACT next step
        elif matched_index == self.current_index + 1:
            self.current_index = matched_index
            self.error_active = False
            self.error_detail = ""
            self.skipped_step_index = None
            self.skip_streak = 0
            return "advanced", self.current_state()

        # Case 2b: From Step 0, user directly places completed 2-legged base (Green + 2 Blue feet)
        elif self.current_index == 0 and matched_index == 2:
            self.current_index = 2
            self.error_active = False
            self.error_detail = ""
            self.skipped_step_index = None
            self.skip_streak = 0
            return "advanced", self.current_state()

        # Case 3: Reverted or temporary hand occlusion - hold current step without resetting progress
        elif matched_index < self.current_index:
            self.skip_streak = 0
            return "holding", f"Verifying [{self.get_step_title(self.current_index)}]"

        # Case 4: Skipped ahead out of order!
        else:
            self.skip_streak += 1
            expected = self.get_step_title(self.current_index + 1)
            detected = self.get_step_title(matched_index)

            # Require at least 2 consecutive consensus cycles to latch sequence error,
            # preventing accidental trigger during in-flight hand placement
            if self.skip_streak >= 2:
                self.error_active = True
                self.skipped_step_index = self.current_index + 1
                self.error_detail = f"SKIPPED STEP! Missing [{expected}], but found [{detected}]"
                return "error", self.error_detail
            else:
                return "holding", f"Verifying [{expected}]..."

    def get_steps_for_hud(self):
        """
        Returns the 9-step sequential inspection checklist for HUD rendering.
        - Steps < current_index are 'verified' ([PASS] in Green)
        - Step == current_index:
          - If error: 'error_current' ([ERR ] in Red)
          - If current_index > 0 or complete: 'current_passed' ([PASS] in Bright Green)
          - If current_index == 0: 'current' ([NOW ] in Amber)
        - Step == current_index + 1: 'next' ([NEXT] in Cyan)
        - Steps > current_index + 1: 'pending' ([    ] in Dim Gray)
        """
        steps = []
        for i, sname in enumerate(self.state_order):
            title = STEP_TITLES.get(sname, sname)
            if i < self.current_index:
                st = "verified"
            elif i == self.current_index:
                if self.error_active:
                    st = "error_current"
                elif self.current_index > 0 or self.is_complete():
                    st = "current_passed"
                else:
                    st = "current"
            elif i == self.current_index + 1:
                st = "next" if not self.error_active else "pending"
            else:
                st = "pending"
            steps.append({"title": title, "state": sname, "status": st, "index": i})
        return steps

    def is_complete(self):
        return self.current_index == len(self.state_order) - 1 and not self.error_active
