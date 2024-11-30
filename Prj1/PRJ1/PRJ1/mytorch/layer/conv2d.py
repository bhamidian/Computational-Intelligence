from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        
        batch_size, _, input_height, input_width = x.shape

        # Calculate output dimensions
        output_height = ((input_height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]) + 1
        output_width = ((input_width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1]) + 1

        # Initialize output tensor
        output = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # Apply padding
        if self.padding != (0, 0):
            x_padded = np.pad(
                x.data,
                pad_width=(
                    (0, 0),  # Batch size
                    (0, 0),  # Channels
                    (self.padding[0], self.padding[0]),  # Height padding
                    (self.padding[1], self.padding[1])   # Width padding
                ),
                mode='constant'
            )
        else:
            x_padded = x.data

        # Perform convolution
        for batch in range(batch_size):
            for out_channel in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Define the region of interest
                        start_i = i * self.stride[0]
                        start_j = j * self.stride[1]
                        end_i = start_i + self.kernel_size[0]
                        end_j = start_j + self.kernel_size[1]

                        # Convolution operation
                        region = x_padded[batch, :, start_i:end_i, start_j:end_j]
                        output[batch, out_channel, i, j] = np.sum(region * self.weight.data[out_channel]) + (
                            self.bias.data[out_channel] if self.need_bias else 0
                        )

        return Tensor(output)
    
    def initialize(self):
        "TODO: initialize weights"
        weight_shape = (self.out_channels, self.in_channels, *self.kernel_size)  # (out_channels, in_channels, height, width)
        self.weight = Tensor(initializer(weight_shape, mode=self.initialize_mode))
        self.grad_weight = np.zeros_like(self.weight.data)

        if self.need_bias:
            bias_shape = (self.out_channels,)
            self.bias = Tensor(initializer(bias_shape, mode="zero"))
            self.grad_bias = np.zeros_like(self.bias.data)

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        
        return [self.weight, self.bias] if self.need_bias else [self.weight]

    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
