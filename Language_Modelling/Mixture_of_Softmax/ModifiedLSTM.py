from config import args as nnargs
from tensorflow.python.ops.rnn_cell_impl import RNNCell,LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
import  tensorflow as tf
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# modification is in class _Liner
class ModifiedLSTM(RNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.

      The default non-peephole implementation is based on:

        http://www.bioinf.jku.at/publications/older/2604.pdf

      S. Hochreiter and J. Schmidhuber.
      "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

      The peephole implementation is based on:

        https://research.google.com/pubs/archive/43905.pdf

      Hasim Sak, Andrew Senior, and Francoise Beaufays.
      "Long short-term memory recurrent neural network architectures for
       large scale acoustic modeling." INTERSPEECH, 2014.

      The class uses optional peep-hole connections, optional cell clipping, and
      an optional projection layer.
      """

    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(ModifiedLSTM, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units
        self._linear1 = None
        self._linear2 = None
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.

        Returns:
          A tuple containing:

          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        if self._linear1 is None:
            scope = vs.get_variable_scope()
            with vs.variable_scope(
                    scope, initializer=self._initializer) as unit_scope:
                if self._num_unit_shards is not None:
                    unit_scope.set_partitioner(
                        partitioned_variables.fixed_size_partitioner(
                            self._num_unit_shards))
                self._linear1 = _Linear([inputs, m_prev], 4 * self._num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = self._linear1([inputs, m_prev])
        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes and not self._w_f_diag:
            scope = vs.get_variable_scope()
            with vs.variable_scope(
                    scope, initializer=self._initializer) as unit_scope:
                with vs.variable_scope(unit_scope):
                    self._w_f_diag = vs.get_variable(
                        "w_f_diag", shape=[self._num_units], dtype=dtype)
                    self._w_i_diag = vs.get_variable(
                        "w_i_diag", shape=[self._num_units], dtype=dtype)
                    self._w_o_diag = vs.get_variable(
                        "w_o_diag", shape=[self._num_units], dtype=dtype)

        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            if self._linear2 is None:
                scope = vs.get_variable_scope()
                with vs.variable_scope(scope, initializer=self._initializer):
                    with vs.variable_scope("projection") as proj_scope:
                        if self._num_proj_shards is not None:
                            proj_scope.set_partitioner(
                                partitioned_variables.fixed_size_partitioner(
                                    self._num_proj_shards))
                        self._linear2 = _Linear(m, self._num_proj, False)
            m = self._linear2(m)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state



class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      self._weights = vs.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)

      # modification here
      self._weights=tf.nn.dropout(self._weights,keep_prob=nnargs.wdrop)
      # modification here

      if build_bias:
        with vs.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
          self._biases = vs.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res