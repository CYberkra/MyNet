"""Interface types and utilities for PGDA model outputs.

Provides PGDAOutput (a dict-like object that also supports tuple unpacking),
make_pgda_output() for constructing it, and unpack_pgda_output() that
handles both PGDAOutput dict objects and legacy tuple returns.
"""


class PGDAOutput:
    """Model output that works as both a dict-like and a tuple-unpackable object.

    Supports all three calling conventions transparently:

        mask, pres, center = output          # tuple unpacking
        output['mask_logits']                 # dict-style key access
        output.mask_logits                     # attribute access

    And the len / bool protocol:
        len(output)  # 3 (or 2 if center_logits is None)
        bool(output) # True
    """

    def __init__(self, mask_logits, presence_logits, center_logits=None):
        self.mask_logits = mask_logits
        self.presence_logits = presence_logits
        self.center_logits = center_logits
        self._iter_items = (mask_logits, presence_logits, center_logits)

    # ----- tuple unpacking -----

    def __iter__(self):
        return iter(self._iter_items)

    def __len__(self):
        return 3

    # ----- dict-style access -----

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._iter_items[key]
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key):
        return key in ('mask_logits', 'presence_logits', 'center_logits')

    def keys(self):
        return ['mask_logits', 'presence_logits', 'center_logits']

    def values(self):
        return list(self._iter_items)

    def items(self):
        return zip(self.keys(), self.values())

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    # ----- representation -----

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mask_logits={self._shape_repr(self.mask_logits)}, "
            f"presence_logits={self._shape_repr(self.presence_logits)}, "
            f"center_logits={self._shape_repr(self.center_logits)})"
        )

    @staticmethod
    def _shape_repr(t):
        if t is None:
            return "None"
        if hasattr(t, 'shape'):
            return f"tensor{tuple(t.shape)}"
        return str(type(t).__name__)


def make_pgda_output(mask_logits, presence_logits, center_logits=None):
    """Construct a PGDAOutput from the three head tensors."""
    return PGDAOutput(mask_logits, presence_logits, center_logits)


def unpack_pgda_output(output):
    """Unpack model output into (mask_logits, presence_logits, center_logits).

    Handles PGDAOutput, plain dict, legacy 3-tuple, and legacy 2-tuple.
    """
    if hasattr(output, 'mask_logits') and hasattr(output, 'presence_logits'):
        return output.mask_logits, output.presence_logits, getattr(output, 'center_logits', None)
    if isinstance(output, dict):
        return output.get('mask_logits'), output.get('presence_logits'), output.get('center_logits', None)
    if isinstance(output, (tuple, list)):
        if len(output) == 3:
            return output[0], output[1], output[2]
        if len(output) == 2:
            return output[0], output[1], None
        raise ValueError(f"Expected tuple of length 2 or 3, got length {len(output)}")
    raise TypeError(f"Expected PGDAOutput, dict, tuple, or list, got {type(output).__name__}")
