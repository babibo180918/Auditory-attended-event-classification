import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, bias):
        """
        ConvLSTM cell.

        Parameters
        ----------
        input_size: int
            Number of expected features in the input.
        hidden_size: int
            Number of features in the hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            If False, then the layer does not use bias weights.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding='same',
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next =  torch.add(torch.mul(f, c_cur), torch.mul(i, g))
        h_next = torch.mul(o, torch.tanh(c_next))        
        del combined, combined_conv, cc_i, cc_f, cc_o, cc_g, h_cur, c_cur
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (Variable(torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device)),
                Variable(torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device)))


class ConvLSTM(nn.Module):
    """ Pytorch Implementation of Convolutional LSTM.
    https://arxiv.org/pdf/1506.04214.pdf
    
    Parameters:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        kernel_size: The size of kernel in convolutions
        num_layers: Number of recurrent layers stacked on each other. Default: 1
        batch_first: Whether or not dimension 0 is the batch or not. Default: True
        dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
        bias: If False, then the layer does not use bias weights. Default: True
        Note: All convolutional connections will use the same padding.

    Input:
        A tensor of size (N, L, input_size, H, W) batch_first = True or (L, N, input_size, H, W) batch_first = False
    Outputs: output, (h_n, c_n)
        output - tensor of shape (N, L, hidden_size, H, W) containing the output features from the last layer of the LSTM, for each element in the sequence.
        h_n - tensor of shape (N, hidden_size, H, W) containing the final hidden state from the last layer of the LSTM for each element in the sequence.
        c_n - tensor of shape (N, hidden_size, H, W) containing the final cell state from the last layer of the LSTM for each element in the sequence.
    Example:
        >>> x = torch.rand((64, 16, 1, 7, 5))
        >>> convlstm = ConvLSTM(1, 32, 3, 1, True, 0.2, True)
        >>> out, (h_n, c_n) = convlstm(x)
    """

    def __init__(self, input_size, hidden_size, kernel_size, num_layers=1,
                 batch_first=True, dropout=0.0, bias=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_size` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_size = self._extend_for_multilayer(hidden_size, num_layers)
        if not len(kernel_size) == len(hidden_size) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        dropout_list = []
        for i in range(0, self.num_layers):
            cur_input_size = self.input_size if i == 0 else self.hidden_size[i - 1]

            cell_list.append(ConvLSTMCell(input_size=cur_input_size,
                                          hidden_size=self.hidden_size[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
            if dropout>0.0 and i<self.num_layers-1:
                dropout_list.append(nn.Dropout(p=dropout))
        self.cell_list = nn.ModuleList(cell_list)
        if len(dropout_list) > 0:
            self.dropout_list = nn.ModuleList(dropout_list)
        else:
            self.dropout_list = None

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (L, N, C, H, W) or (N, L, C, H, W)
        hidden_state: None.

        Returns
        -------
        layer_output, (h, c)
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            if self.dropout_list is not None and layer_idx<self.num_layers-1:
                layer_output = self.dropout_list[layer_idx](layer_output)
            cur_layer_input = layer_output
            del output_inner
            
        del cur_layer_input, hidden_state 
        return layer_output, (h, c)

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param