"""Coarse client-local progress only; no examples, RNG calls or model updates."""
import math
import time


def duration_text(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class BatchProgress:
    """Print at stage start, first completed batch, time intervals and completion.

    Updates occur after completed batches; this is not a background heartbeat
    while a single long batch is executing. ETA is a phase-local estimate based
    on observed example throughput, not an estimate for the whole FL experiment.
    An interval of zero disables this reporter completely.
    """
    def __init__(self, label, *, total_batches, total_examples, epochs=1,
                 interval_seconds=30, clock=None, emit=None):
        if (type(interval_seconds) not in (int, float) or not math.isfinite(interval_seconds)
                or interval_seconds < 0):
            raise ValueError('progress interval must be finite and nonnegative')
        if min(total_batches, total_examples, epochs) < 1 or total_batches % epochs:
            raise ValueError('progress requires positive totals and equal batches per epoch')
        self.label, self.total_batches, self.total_examples, self.epochs = label, total_batches, total_examples, epochs
        self.interval = interval_seconds
        self.clock = clock or time.perf_counter
        self.emit = emit or (lambda message: print(message, flush=True))
        self.started = self.last_printed = self.clock()
        self.batches = self.examples = self.loss_examples = 0
        self.loss_sum = 0.0
        if self.interval:
            self.emit(f'[{label}] START: {total_examples:,} examples; {epochs} epoch(s); '
                      f'{total_batches:,} batches. Progress after completed batches every ~{interval_seconds:g}s.')

    def update(self, examples, *, loss=None):
        self.batches += 1
        self.examples += examples
        if self.batches > self.total_batches or self.examples > self.total_examples:
            raise ValueError('progress exceeds declared phase totals')
        if loss is not None:
            self.loss_sum += loss*examples
            self.loss_examples += examples
        if not self.interval:
            return
        now = self.clock()
        final = self.batches == self.total_batches
        if not (self.batches == 1 or final or now-self.last_printed >= self.interval):
            return
        elapsed = max(0.0, now-self.started)
        rate = self.examples/elapsed if elapsed else 0.0
        eta = duration_text((self.total_examples-self.examples)/rate) if rate else 'pending'
        epoch = min(self.epochs, (self.batches-1)//(self.total_batches//self.epochs)+1)
        loss_text = f'; running loss={self.loss_sum/self.loss_examples:.4f}' if self.loss_examples else ''
        self.emit(f'[{self.label}] {"DONE" if final else "PROGRESS"}: epoch {epoch}/{self.epochs}; '
                  f'batch {self.batches:,}/{self.total_batches:,} ({100*self.batches/self.total_batches:.1f}%); '
                  f'examples {self.examples:,}/{self.total_examples:,}{loss_text}; '
                  f'elapsed={duration_text(elapsed)}; {rate:.1f} examples/s; phase ETA~{eta}')
        self.last_printed = now
