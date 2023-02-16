from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 4356000,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 2,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        print(f'x.shape: {x.shape}')
        batch_size, channels, height, width = x.size()

        # (batch, 3, height, width) -> (batch, 3*height*width)
        x = x.view(batch_size, -1)
        x = x.to(self.model[0].weight.dtype)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()