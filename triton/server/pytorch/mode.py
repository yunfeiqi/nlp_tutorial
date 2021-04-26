import torch


class MyDecisionGate(torch.nn.Module):
    def forward(self, INPUT_0):
        if INPUT_0.sum() > 0:
            return INPUT_0
        else:
            return -INPUT_0


class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, INPUT_0, INPUT_1):
        OUTPUT_0 = torch.tanh(self.dg(self.linear(INPUT_0)) + INPUT_1)
        OUTPUT_1 = OUTPUT_0
        return OUTPUT_0, OUTPUT_1


my_cell = MyCell(MyDecisionGate())
scripted_gate = torch.jit.script(my_cell)

scripted_gate.save("model.pt")

x, h = torch.rand(3, 4), torch.rand(3, 4)
print(scripted_gate(x, h))
