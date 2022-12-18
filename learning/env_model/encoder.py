


'''
class VariableEncoder(BaseNN):
    def __init__(self, context: Context):
        super().__init__(context)
        
        self.sub_modules: Dict[str, nn.Module] = {}

        for var in self.env.names_input:
            d_in = self.v(var).size
            d_h = self.dims.variable_encoder_hidden
            d_out = self.dims.variable_encoding

            self.sub_modules[var] = nn.Sequential(
                nn.Linear(d_in, d_h, **self.torchargs),
                nn.PReLU(d_h, **self.torchargs),
                nn.Linear(d_h, d_h, **self.torchargs),
                nn.LeakyReLU(),
                nn.Linear(d_h, d_out, **self.torchargs))

        for key, linear in self.sub_modules.items():
            self.add_module(f"{key}_encoder", linear)

    def forward_all(self, data: Batch):
        out = Batch(data.n)
        for var in self.sub_modules.keys():
            if var in data:
                out[var] = self.forward(var, data[var])
        return out

    def forward(self, var: str, data: torch.Tensor) -> torch.Tensor:
        sub_module = self.sub_modules[var]
        return sub_module(data)
'''


