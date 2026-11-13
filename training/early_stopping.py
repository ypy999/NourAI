import math


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, enabled=True):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer")
        if not math.isfinite(float(min_delta)) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        self.patience, self.min_delta, self.enabled = patience, float(min_delta), enabled
        self.best_loss = None
        self.bad_epochs = 0

    def update(self, loss):
        if loss is None or not math.isfinite(float(loss)):
            raise ValueError("early stopping requires a finite validation loss")
        improved = self.best_loss is None or loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss, self.bad_epochs = float(loss), 0
        else:
            self.bad_epochs += 1
        return improved, self.enabled and self.bad_epochs >= self.patience

    def state_dict(self):
        return {"best_loss": self.best_loss, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state):
        self.best_loss = state["best_loss"]
        self.bad_epochs = int(state["bad_epochs"])
