"""Interface types and utilities for PGDA model outputs.

Provides PGDAOutput (a dict-like object that also supports tuple unpacking),
GprMambaSepOutput (extends PGDAOutput with A_hat, S_hat, G_hat component fields),
make_pgda_output() for constructing it, and unpack_model_output() that
handles PGDAOutput, GprMambaSepOutput, plain dict, and legacy tuple returns.
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


class GprMambaSepOutput(PGDAOutput):
    """Model output for GprMambaSep with separable A/S/G component fields.

    Extends PGDAOutput with A_hat (air wave), S_hat (surface reflection),
    G_hat (geological signal), and optional v2.1 curve-picking heads.
    Supports all three calling conventions transparently:

        mask, pres, center, A_hat, S_hat, G_hat = output  # tuple unpacking
        output['A_hat']                                     # dict-style key access
        output.G_hat                                         # attribute access

    The `__contains__`, `keys()`, `get()` etc. all include the six fields.
    Backward-compatible aliases `G_mask_logits`, `G_presence_logits`, and
    `G_center_logits` map to the standard three PGDA head tensors.
    """

    _ALIASES = {
        'G_mask_logits': 'mask_logits',
        'G_presence_logits': 'presence_logits',
        'G_center_logits': 'center_logits',
    }

    def __init__(self, mask_logits, presence_logits, center_logits=None,
                 A_hat=None, S_hat=None, G_hat=None, component_gates=None,
                 curve_logits=None, global_no_target_logits=None,
                 uncertainty_logits=None):
        super().__init__(mask_logits, presence_logits, center_logits)
        self.A_hat = A_hat
        self.S_hat = S_hat
        self.G_hat = G_hat
        # Optional diagnostic tensor, shape (B, 3, H, W).  It is deliberately
        # not part of tuple unpacking so old six-value callers remain valid.
        self.component_gates = component_gates
        # Optional v2.1 curve-picking fields.  They are not part of tuple
        # unpacking so legacy training/eval callers remain valid.
        self.curve_logits = curve_logits
        self.global_no_target_logits = global_no_target_logits
        self.uncertainty_logits = uncertainty_logits
        self._iter_items = (mask_logits, presence_logits, center_logits,
                            A_hat, S_hat, G_hat)

    def __len__(self):
        return 6

    def __getitem__(self, key):
        if isinstance(key, str) and key in self._ALIASES:
            key = self._ALIASES[key]
        return super().__getitem__(key)

    def __contains__(self, key):
        return key in ('mask_logits', 'presence_logits', 'center_logits',
                       'A_hat', 'S_hat', 'G_hat', 'component_gates',
                       'curve_logits', 'global_no_target_logits',
                       'uncertainty_logits', *self._ALIASES.keys())

    def keys(self):
        return ['mask_logits', 'presence_logits', 'center_logits',
                'A_hat', 'S_hat', 'G_hat', 'component_gates',
                'curve_logits', 'global_no_target_logits', 'uncertainty_logits']

    def values(self):
        # Dict-like views should include diagnostics and v2.1 curve heads.
        # Tuple unpacking intentionally remains six items via _iter_items/.__iter__.
        return [self.mask_logits, self.presence_logits, self.center_logits,
                self.A_hat, self.S_hat, self.G_hat, self.component_gates,
                self.curve_logits, self.global_no_target_logits,
                self.uncertainty_logits]

    def items(self):
        return zip(self.keys(), self.values())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mask_logits={self._shape_repr(self.mask_logits)}, "
            f"presence_logits={self._shape_repr(self.presence_logits)}, "
            f"center_logits={self._shape_repr(self.center_logits)}, "
            f"A_hat={self._shape_repr(self.A_hat)}, "
            f"S_hat={self._shape_repr(self.S_hat)}, "
            f"G_hat={self._shape_repr(self.G_hat)}, "
            f"component_gates={self._shape_repr(self.component_gates)}, "
            f"curve_logits={self._shape_repr(self.curve_logits)}, "
            f"global_no_target_logits={self._shape_repr(self.global_no_target_logits)}, "
            f"uncertainty_logits={self._shape_repr(self.uncertainty_logits)})"
        )


class AeroPathOutput(PGDAOutput):
    """Structured interface-picking output for AeroPath-SSD.

    The standard three logits preserve compatibility with the existing trainer.
    ``curve_logits`` are unary interface energies and ``path_marginals`` are
    the differentiable dynamic-programming path distribution over time.
    """

    def __init__(self, mask_logits, presence_logits, center_logits, *, curve_logits,
                 path_marginals, no_pick_logits, air_reduced_input,
                 uncertainty_logits=None):
        super().__init__(mask_logits, presence_logits, center_logits)
        self.curve_logits = curve_logits
        self.path_marginals = path_marginals
        self.no_pick_logits = no_pick_logits
        self.air_reduced_input = air_reduced_input
        # Log-variance map used only by the structured path uncertainty loss.
        # It is deliberately separate from the mask and curve probabilities.
        self.uncertainty_logits = uncertainty_logits
        # Keep the established global-head spelling available to generic tooling.
        self.global_no_target_logits = no_pick_logits

    def __contains__(self, key):
        return key in {
            "mask_logits", "presence_logits", "center_logits", "curve_logits",
            "path_marginals", "no_pick_logits", "air_reduced_input",
            "uncertainty_logits", "global_no_target_logits",
        }

    def keys(self):
        return [
            "mask_logits", "presence_logits", "center_logits", "curve_logits",
            "path_marginals", "no_pick_logits", "air_reduced_input",
            "uncertainty_logits", "global_no_target_logits",
        ]

    def values(self):
        return [
            self.mask_logits, self.presence_logits, self.center_logits,
            self.curve_logits, self.path_marginals, self.no_pick_logits,
            self.air_reduced_input, self.uncertainty_logits,
            self.global_no_target_logits,
        ]


def make_gprmambasep_output(mask_logits, presence_logits, center_logits=None,
                             A_hat=None, S_hat=None, G_hat=None, component_gates=None,
                             curve_logits=None, global_no_target_logits=None,
                             uncertainty_logits=None):
    """Construct a GprMambaSepOutput from the head/component tensors."""
    return GprMambaSepOutput(mask_logits, presence_logits, center_logits,
                              A_hat, S_hat, G_hat, component_gates=component_gates,
                              curve_logits=curve_logits,
                              global_no_target_logits=global_no_target_logits,
                              uncertainty_logits=uncertainty_logits)


def unpack_pgda_output(output):
    """Unpack model output into (mask_logits, presence_logits, center_logits).

    Handles PGDAOutput, GprMambaSepOutput, plain dict, legacy 3-tuple, and
    legacy 2-tuple.
    """
    if hasattr(output, 'mask_logits') and hasattr(output, 'presence_logits'):
        return (output.mask_logits, output.presence_logits,
                getattr(output, 'center_logits', None))
    if isinstance(output, dict):
        return (output.get('mask_logits'), output.get('presence_logits'),
                output.get('center_logits', None))
    if isinstance(output, (tuple, list)):
        if len(output) >= 3:
            return output[0], output[1], output[2]
        if len(output) == 2:
            return output[0], output[1], None
        raise ValueError(f"Expected tuple of length 2 or 3, got length {len(output)}")
    raise TypeError(f"Expected PGDAOutput, dict, tuple, or list, got {type(output).__name__}")


def unpack_model_output(output):
    """Unpack model output into the standard three logit tensors.

    Identical to unpack_pgda_output; provided as a more generic name.
    """
    return unpack_pgda_output(output)
