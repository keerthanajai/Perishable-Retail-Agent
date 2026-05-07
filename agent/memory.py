from typing import List, Dict


class ConversationMemory:
    """
    Stores the full conversation history so the agent can handle
    follow-up questions like 'now do that for store 5' or 'why?'
    """

    def __init__(self, max_turns: int = 20):
        self.history: List[Dict] = []
        self.max_turns = max_turns
        self.last_context: Dict = {}  

    def add_user(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def get_history(self) -> List[Dict]:
        return self.history

    def update_context(self, **kwargs):
        """Remember last used store_id, item_id, date for follow-ups."""
        self.last_context.update({k: v for k, v in kwargs.items() if v is not None})

    def get_context(self) -> Dict:
        return self.last_context

    def _trim(self):
        """Keep only the last N turns to avoid exceeding context window."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def clear(self):
        self.history = []
        self.last_context = {}

    def __len__(self):
        return len(self.history)