from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        # Get input dimensions
        batch_size, channels, input_height, input_width = x.shape

        # Calculate output dimensions
        output_height = ((input_height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]) + 1
        output_width = ((input_width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1]) + 1

        # Initialize output tensor
        output = np.zeros((batch_size, channels, output_height, output_width))

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

        # Perform average pooling
        for batch in range(batch_size):
            for channel in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Define the region of interest
                        start_i = i * self.stride[0]
                        start_j = j * self.stride[1]
                        end_i = start_i + self.kernel_size[0]
                        end_j = start_j + self.kernel_size[1]

                        # Pooling operation
                        region = x_padded[batch, channel, start_i:end_i, start_j:end_j]
                        output[batch, channel, i, j] = np.mean(region)

        return Tensor(output)
    
    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
