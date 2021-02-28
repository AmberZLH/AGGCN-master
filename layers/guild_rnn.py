from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from layers import nn



class LegacyGRUCell(torch.nn.RNNCell):
    """ Groundhog's implementation of GRUCell
    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, *args, **kwargs):
        super(LegacyGRUCell, self).__init__(*args, **kwargs)


    def __call__(self, inputs, aspect_input, params, state, scope=None):

        if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

        all_inputs = list(inputs) + [state]
        r = torch.nn.sigmoid(nn.linear(all_inputs, self._num_units, False, False,
                                     scope="reset_gate"))

        u = torch.nn.sigmoid(nn.linear(all_inputs, self._num_units, False, False,
                                     scope="update_gate"))
        all_inputs = list(inputs) + [r * state]

        c = torch.tanh(nn.linear(all_inputs, self._num_units, True, False,
                               scope="candidate"))

        if self._keep_prob and self._keep_prob < 1.0:
                c = torch.nn.dropout(c, self._keep_prob)

        new_state = (1.0 - u) * state + u * c

        rh = torch.nn.sigmoid(nn.linear(new_state, self._num_units, False, False,
                                      scope="reset_gate1"))
        uh = torch.nn.sigmoid(nn.linear(new_state, self._num_units, False, False,
                                      scope="update_gate1"))
        ch = torch.tanh(rh * nn.linear(new_state, self._num_units, True, False,
                                     scope="candidate1"))
        if self._keep_prob and self._keep_prob < 1.0:
            ch = torch.nn.dropout(ch, self._keep_prob)

        new_state2 = (1.0 - uh) * new_state + uh * ch

        rh = torch.nn.sigmoid(nn.linear(new_state2, self._num_units, False, False,
                                      scope="reset_gate2"))
        uh = torch.nn.sigmoid(nn.linear(new_state2, self._num_units, False, False,
                                      scope="update_gate2"))
        ch = torch.tanh(rh * nn.linear(new_state2, self._num_units, True, False,
                                     scope="candidate2"))
        if self._keep_prob and self._keep_prob < 1.0:
            ch = torch.nn.dropout(ch, self._keep_prob)

        new_state3 = (1.0 - uh) * new_state2 + uh * ch

        rh = torch.nn.sigmoid(nn.linear(new_state3, self._num_units, False, False,
                                      scope="reset_gate3"))
        uh = torch.nn.sigmoid(nn.linear(new_state3, self._num_units, False, False,
                                      scope="update_gate3"))
        ch = torch.tanh(rh * nn.linear(new_state3, self._num_units, True, False,
                                     scope="candidate3"))
        if self._keep_prob and self._keep_prob < 1.0:
            ch = torch.nn.dropout(ch, self._keep_prob)

        new_state4 = (1.0 - uh) * new_state3 + uh * ch

        rh = torch.nn.sigmoid(nn.linear(new_state4, self._num_units, False, False,
                                      scope="reset_gate4"))
        uh = torch.nn.sigmoid(nn.linear(new_state4, self._num_units, False, False,
                                      scope="update_gate4"))
        ch = torch.tanh(rh * nn.linear(new_state4, self._num_units, True, False,
                                     scope="candidate4"))
        if self._keep_prob and self._keep_prob < 1.0:
            ch = torch.nn.dropout(ch, self._keep_prob)

        new_state5 = (1.0 - uh) * new_state4 + uh * ch

        return new_state5, new_state5

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


class DL4MTGRULAUTransiLNCell(torch.nn.RNNCell):
    """ DL4MT's implementation of GRUCell with LAU and Transition
    Args:
        num_units: int, The number of units in the RNN cell.
        reuse: (optional) Python boolean describing whether to reuse
            variables in an existing scope.  If not `True`, and the existing
            scope already has the given variables, an error is raised.
    """

    def __init__(self, num_transi, num_units, keep_prob=None, reuse=None):
        super(DL4MTGRULAUTransiLNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._keep_prob = keep_prob
        self._num_transi = num_transi

    def __call__(self, inputs, aspect_input, params, state, scope=None):
        with torch.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state, aspect_input]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            if not isinstance(aspect_input, (list, tuple)):
                aspect_input = [aspect_input]

            all_inputs = list(inputs) + [state]

            r = torch.nn.sigmoid(nn.LayerNorm(nn.linear(all_inputs, self._num_units, False, False,
                                                scope="reset_gate"),
                                         scope="reset_gate_ln"))
            r2 = torch.nn.sigmoid(nn.LayerNorm(nn.linear(all_inputs, self._num_units, False, False,
                                                 scope="reset_gate2"),
                                          scope="reset_gate2_ln"))
            u = torch.nn.sigmoid(nn.LayerNorm(nn.linear(all_inputs, self._num_units, False, False,
                                                scope="update_gate"),
                                         scope="update_gate_ln"))

            nn.linear_state = nn.linear(state, self._num_units, True, False, scope="nn.linear_state")
            nn.linear_inputs = nn.linear(inputs, self._num_units, False, False, scope="nn.linear_inputs")
            nn.linear_inputs_transform_l = nn.linear(inputs, self._num_units, False, False, scope="nn.linear_inputs_transform_l")
            nn.linear_inputs_transform_a = nn.linear(inputs, self._num_units, False, False, scope="nn.linear_inputs_transform_a")
            if params.use_aspect_gate:
                aspect_source_input = list(aspect_input) + [state]
                aspect = torch.nn.relu(nn.LayerNorm(nn.linear(aspect_source_input, self._num_units, False, False,
                                                      scope="aspect_gate"),
                                               scope="aspect_gate_ln"))
                c = torch.tanh(
                    aspect * nn.linear_inputs + r * nn.linear_state) + aspect * nn.linear_inputs_transform_a + r2 * nn.linear_inputs_transform_l
            else:
                c = torch.tanh(nn.linear_inputs + r * nn.linear_state) + r2 * nn.linear_inputs_transform_l
            if self._keep_prob and self._keep_prob < 1.0:
                c = torch.nn.dropout(c, self._keep_prob)

            new_state = (1.0 - u) * state + u * c

            for i in range(int(self._num_transi)):
                rh = torch.nn.sigmoid(nn.LayerNorm(nn.linear(new_state, self._num_units, False, False,
                                                     scope="trans_reset_gate_l%d" % i),
                                              scope="trans_reset_gate_ln_l%d" % i))
                uh = torch.nn.sigmoid(nn.LayerNorm(nn.linear(new_state, self._num_units, False, False,
                                                     scope="trans_update_gate_l%d" % i),
                                              scope="trans_update_gate_ln_l%d" % i))
                ch = torch.tanh(rh * nn.linear(new_state, self._num_units, True, False,
                                         scope="trans_candidate_l%d" % i))
                if self._keep_prob and self._keep_prob < 1.0:
                    ch = torch.nn.dropout(ch, self._keep_prob)

                new_state = (1.0 - uh) * new_state + uh * ch
            return new_state, new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units