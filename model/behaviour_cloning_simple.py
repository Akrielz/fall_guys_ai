from torch import nn


class BehaviorCloning(nn.Module):
    def __init__(
            self,
            in_channels = 3,
            channels_conv3d = 64,
    ):
        super(BehaviorCloning, self).__init__()

    def forward(self, x):
        # x is shape [batch_size, num_channels, time, height, width]


        pass



def main():
    pass


if __name__ == "__main__":
    main()